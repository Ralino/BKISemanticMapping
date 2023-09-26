#include <algorithm>
#include <assert.h>

#include "semantic_bki/mapping/bkioctree_node.h"

namespace semantic_bki {

    /// Default static values
    int Semantics::num_class = 2;
    float Semantics::sf2 = 1.0f;
    float Semantics::ell = 1.0f;
    float Semantics::prior = 0.5f;
    float Semantics::var_thresh = 1000.0f;
    float Semantics::free_thresh = 0.3f;
    float Semantics::occupied_thresh = 0.7f;

    void Semantics::get_probs(std::vector<float>& probs) const {
      assert (probs.size() == num_class);
      float sum = 0;
      for (auto m : ms)
        sum += m;
      for (int i = 0; i < num_class; ++i)
        probs[i] = ms[i] / sum;
    }

    void Semantics::get_vars(std::vector<float>& vars) const {
      assert (vars.size() == num_class);
      float sum = 0;
      for (auto m : ms)
        sum += m;
      for (int i = 0; i < num_class; ++i)
        vars[i] = ((ms[i] / sum) - (ms[i] / sum) * (ms[i] / sum)) / (sum + 1);
    }
}
