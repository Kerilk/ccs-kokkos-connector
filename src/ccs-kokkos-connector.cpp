#include <impl/Kokkos_Profiling_Interface.hpp>
#include <cconfigspace.h>
#include <string>
#include <iostream>
#include <cassert>
#include <map>

#define CCS_CHECK(expr)            \
  do {                             \
    assert(CCS_SUCCESS == (expr)); \
  } while(0)

constexpr const double epsilon = 1E-24;

using namespace Kokkos::Tools::Experimental;

/* from Apollo Kokkos Connector, don't ask me... */
static size_t num_unconverged_regions;

Kokkos::Tools::Experimental::ToolProgrammingInterface helper_functions;
void invoke_fence(uint32_t devID) {
  if ((helper_functions.fence != nullptr) && (num_unconverged_regions > 0)) {
    helper_functions.fence(devID);
  }
}

extern "C" void kokkosp_provide_tool_programming_interface(
    uint32_t num_actions,
    Kokkos::Tools::Experimental::ToolProgrammingInterface action) {
  helper_functions = action;
}

extern "C" void kokkosp_request_tool_settings(
    uint32_t num_responses,
    Kokkos::Tools::Experimental::ToolSettings *response) {
  response->requires_global_fencing = false;
}

extern "C" void kokkosp_begin_parallel_for(const char *name,
                                           const uint32_t devID,
                                           uint64_t *kID) {
  invoke_fence(devID);
  *kID = devID;
}

extern "C" void kokkosp_end_parallel_for(const uint64_t kID) {
  uint32_t devID = kID;
  invoke_fence(devID);
}

extern "C" void kokkosp_begin_parallel_scan(const char *name,
                                            const uint32_t devID,
                                            uint64_t *kID) {
  invoke_fence(devID);
  *kID = devID;
}

extern "C" void kokkosp_end_parallel_scan(const uint64_t kID) {
  uint32_t devID = kID;
  invoke_fence(devID);
}

extern "C" void kokkosp_begin_parallel_reduce(const char *name,
                                              const uint32_t devID,
                                              uint64_t *kID) {
  invoke_fence(devID);
  *kID = devID;
}

extern "C" void kokkosp_end_parallel_reduce(const uint64_t kID) {
  uint32_t devID = kID;
  invoke_fence(devID);
}
/* ...end blind copy pasting... */

extern "C" void kokkosp_parse_args(int argc, char **argv) {
}

extern "C" void kokkosp_print_help(char *) {
  std::string OPTIONS_BLOCK =
      R"(
CConfigSpace connector for Kokkos, supported options

--autotuner=[string] : chose autotuner to use, supported values: 'random', default: 'random'
)";

  std::cout << OPTIONS_BLOCK;
}

extern "C" void kokkosp_init_library(const int loadSeq,
                                     const uint64_t interfaceVer,
                                     const uint32_t devInfoCount,
                                     void *deviceInfo) {
  std::cout << "Initializing CConfigSpace adapter" << std::endl;
  CCS_CHECK(ccs_init());
}

extern "C" void kokkosp_finalize_library() {
  std::cout << "Finalizing CConfigSpace adapter" << std::endl;
  CCS_CHECK(ccs_fini());
}


static ccs_hyperparameter_t variable_info_to_hyperparameter(
    const char *name,
    VariableInfo *info) {
  ccs_hyperparameter_t ret;
  switch (info->valueQuantity) {
  case CandidateValueType::kokkos_value_set:
    {
      ccs_datum_t *values = new ccs_datum_t[info->candidates.set.size];
      for (size_t i = 0; i < info->candidates.set.size; i++) {
        switch (info->type) {
        case ValueType::kokkos_value_double:
          values[i] = ccs_float(info->candidates.set.values.double_value[i]);
          break;
        case ValueType::kokkos_value_int64:
          values[i] = ccs_int(info->candidates.set.values.int_value[i]);
          break;
        case ValueType::kokkos_value_string:
          values[i] = ccs_string(info->candidates.set.values.string_value[i]);
          break;
        default:
          assert(false && "Unknown ValueType");
        }
      }
      switch (info->category) {
      case StatisticalCategory::kokkos_value_categorical:
        CCS_CHECK(ccs_create_categorical_hyperparameter(
          name, info->candidates.set.size, values, 0, NULL, &ret));
        break;
      case StatisticalCategory::kokkos_value_ordinal:
        CCS_CHECK(ccs_create_ordinal_hyperparameter(
          name, info->candidates.set.size, values, 0, NULL, &ret));
        break;
      case StatisticalCategory::kokkos_value_interval:
      case StatisticalCategory::kokkos_value_ratio:
        CCS_CHECK(ccs_create_discrete_hyperparameter(
          name, info->candidates.set.size, values, 0, NULL, &ret));
        break;
      default:
        assert(false && "Unknown StatisticalCategory");
      }
      delete[] values;
    }
    break;
  case CandidateValueType::kokkos_value_range:
    switch (info->type) {
    case ValueType::kokkos_value_double:
      {
        ccs_float_t lower = info->candidates.range.lower.double_value;
        ccs_float_t upper = info->candidates.range.upper.double_value;
        ccs_float_t step = info->candidates.range.step.double_value;
        assert(0.0 <= step);
        if (info->candidates.range.openLower) {
          if (step == 0.0)
            lower = lower + epsilon;
          else
            lower = lower + step;
        }
        if (!info->candidates.range.openUpper) {
          if (step == 0.0)
            upper = upper + epsilon;
          else // this is dubious/would require verification
            upper = upper + step;
        }
        CCS_CHECK(ccs_create_numerical_hyperparameter(
          name, CCS_NUM_FLOAT, lower, upper, step, lower, NULL, &ret));
      }
      break;
    case ValueType::kokkos_value_int64:
      {
        ccs_int_t lower = info->candidates.range.lower.int_value;
        ccs_int_t upper = info->candidates.range.upper.int_value;
        ccs_int_t step = info->candidates.range.step.int_value;
        assert(0 <= step);
        if (info->candidates.range.openLower) {
          if (step == 0)
            lower = lower + 1;
          else
            lower = lower + step;
        }
        if (!info->candidates.range.openUpper) {
          if (step == 0)
            upper = upper + epsilon;
          else // this is dubious/would require verification
            upper = upper + step;
        }
        CCS_CHECK(ccs_create_numerical_hyperparameter(
          name, CCS_NUM_INTEGER, lower, upper, step, lower, NULL, &ret));
      }
      break;
    default:
      assert(false && "Invalid ValueType");
    }
    break;
  case CandidateValueType::kokkos_value_unbounded:
    switch (info->type) {
    case ValueType::kokkos_value_double:
      {
        ccs_float_t lower = -CCS_INFINITY;
        ccs_float_t upper = CCS_INFINITY;
        ccs_float_t step = 0.0;
        CCS_CHECK(ccs_create_numerical_hyperparameter(
          name, CCS_NUM_FLOAT, lower, upper, step, lower, NULL, &ret));
      }
      break;
    case ValueType::kokkos_value_int64:
      {
        ccs_int_t lower = CCS_INT_MIN;
        ccs_int_t upper = CCS_INT_MAX;
        ccs_int_t step = 0;
        CCS_CHECK(ccs_create_numerical_hyperparameter(
          name, CCS_NUM_INTEGER, lower, upper, step, lower, NULL, &ret));
      }
      break;
    case ValueType::kokkos_value_string:
        CCS_CHECK(ccs_create_string_hyperparameter(
          name, NULL, &ret));
      break;
    default:
      assert(false && "Invalid ValueType");
    }
    break;
  default:
    assert(false && "Unknown CandidateValueType");
  }
  return ret;
}

std::map<size_t, ccs_hyperparameter_t> features;

extern "C" void
kokkosp_declare_input_type(const char *name, const size_t id,
                           Kokkos::Tools::Experimental::VariableInfo *info) {
  features[id] = variable_info_to_hyperparameter(name, info);
}

std::map<size_t, ccs_hyperparameter_t> hyperparameters;

extern "C" void
kokkosp_declare_output_type(const char *name, const size_t id,
                            Kokkos::Tools::Experimental::VariableInfo *info) {
  hyperparameters[id] = variable_info_to_hyperparameter(name, info);
}
