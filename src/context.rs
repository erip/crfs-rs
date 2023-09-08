use bitflags::bitflags;
use ndarray::prelude::*;

bitflags! {
    /// Functionality flags for contexts
    #[derive(Default)]
    pub struct Flag: u32 {
        const BASE = 0x01;
        const VITERBI = 0x01;
        const MARGINALS = 0x02;
        const ALL = 0xFF;
    }
}

bitflags! {
    /// Reset flags
    pub struct Reset: u32 {
        /// Reset state scores
        const STATE = 0x01;
        /// Reset transition scores
        const TRANS = 0x02;
        /// Reset all
        const ALL = 0xFF;
    }
}

/// Context maintains internal data for an instance
#[derive(Debug, Clone, Default)]
pub struct Context {
    /// Flag specifying the functionality
    flag: Flag,
    /// The total number of distinct labels
    pub num_labels: u32,
    /// The number of items in the instance
    pub num_items: u32,
    /// The maximum number of labels
    cap_items: u32,
    /// Logarithm of the normalization factor for the instance.
    ///
    /// This is equivalent to the total scores of all paths in the lattice.
    log_norm: f64,
    /// State scores
    ///
    /// This is a `[T][L]` matrix whose element `[t][l]` presents total score
    /// of state features associating label #l at #t.
    pub state: Array<f64, Ix1>,
    /// Transition scores
    ///
    /// This is a `[L][L]` matrix whose element `[i][j]` represents the total
    /// score of transition features associating labels #i and #j.
    pub trans: Array<f64, Ix1>,
    /// Alpha score matrix
    ///
    /// This is a `[T][L]` matrix whose element `[t][l]` presents the total
    /// score of paths starting at BOS and arriving at (t, l).
    alpha_score: Array<f64, Ix1>,
    /// Beta score matrix
    ///
    /// This is a `[T][L]` matrix whose element `[t][l]` presents the total
    /// score of paths starting at (t, l) and arriving at EOS.
    beta_score: Array<f64, Ix1>,
    /// Scale factor vector
    ///
    /// This is a `[T]` vector whose element `[t]` presents the scaling
    /// coefficient for the alpha_score and beta_score.
    scale_factor: Array<f64, Ix1>,
    /// Row vector (work space)
    ///
    /// This is a `[T]` vector used internally for a work space.
    row: Array<f64, Ix1>,
    /// Backward edges
    ///
    /// This is a `[T][L]` matrix whose element `[t][j]` represents the label #i
    /// that yields the maximum score to arrive at (t, j).
    /// This member is available only with `CTXF_VITERBI` flag enabled.
    backward_edge: Array<u32, Ix1>,
    /// Exponents of state scores
    ///
    /// This is a `[T][L]` matrix whose element `[t][l]` presents the exponent
    /// of the total score of state features associating label #l at #t.
    /// This member is available only with `CTXF_MARGINALS` flag.
    exp_state: Array<f64, Ix1>,
    /// Exponents of transition scores.
    ///
    /// This is a `[L][L]` matrix whose element `[i][j]` represents the exponent
    /// of the total score of transition features associating labels #i and #j.
    /// This member is available only with `CTXF_MARGINALS` flag.
    exp_trans: Array<f64, Ix1>,
    /// Model expectations of states.
    ///
    /// This is a `[T][L]` matrix whose element `[t][l]` presents the model
    /// expectation (marginal probability) of the state (t,l)
    /// This member is available only with CTXF_MARGINALS flag.
    mexp_state: Array<f64, Ix1>,
    /// Model expectations of transitions.
    ///
    /// This is a `[L][L]` matrix whose element `[i][j]` presents the model
    /// expectation of the transition (i--j).
    /// This member is available only with `CTXF_MARGINALS` flag.
    mexp_trans: Array<f64, Ix1>,
}

impl Context {
    pub fn new(flag: Flag, l: u32, t: u32) -> Self {
        let l = l as usize;
        let trans = Array::zeros(l*l);
        let (exp_trans, mexp_trans) = if flag.contains(Flag::MARGINALS) {
            (Array::zeros(l * l + 4), Array::zeros(l*l))
        } else {
            (Array::zeros(1), Array::zeros(1))
        };
        let mut ctx = Self {
            flag,
            trans,
            exp_trans,
            mexp_trans,
            num_items: 0,
            num_labels: l as u32,
            ..Default::default()
        };
        ctx.set_num_items(t);
        // t gives the 'hint' for maximum length of items.
        ctx.num_items = 0;
        ctx
    }

    pub fn set_num_items(&mut self, t: u32) {
        self.num_items = t;
        if self.cap_items < t {
            let l = self.num_labels as usize;
            let t = t as usize;
            self.alpha_score = Array::zeros(t * l);
            self.beta_score = Array::zeros(t * l);
            self.scale_factor = Array::zeros(t);
            self.row = Array::zeros(l);
            if self.flag.contains(Flag::VITERBI) {
                self.backward_edge = Array::zeros(t * l);
            }
            self.state = Array::zeros(t * l);
            if self.flag.contains(Flag::MARGINALS) {
                self.exp_state = Array::zeros(t * l + 4);
                self.mexp_state = Array::zeros(t * l);
            }
            self.cap_items = t as u32;
        }
    }

    pub fn reset(&mut self, flag: Reset) {
        let t = self.num_items as usize;
        let l = self.num_labels as usize;
        if flag.contains(Reset::STATE) {
            self.state.slice_mut(s![..t * l]).fill(0.0);
        }
        if flag.contains(Reset::TRANS) {
            self.trans.slice_mut(s![..l * l]).fill(0.0);
        }
        if self.flag.contains(Flag::MARGINALS) {
            self.mexp_state.slice_mut(s![..t * l]).fill(0.0);
            self.mexp_trans.slice_mut(s![..l * l]).fill(0.0);
            self.log_norm = 0.0;
        }
    }

    pub fn exp_transition(&mut self) {
        let l = self.num_labels as usize;
        self.exp_trans.slice_mut(s![..l * l]).assign(&self.trans);
        for i in 0..(l * l) {
            self.exp_trans[i] = self.exp_trans[i].exp();
        }
    }

    pub fn viterbi(&mut self) -> (Vec<u32>, f64) {
        let mut score;
        let l = self.num_labels as usize;
        // Compute the scores at (0, *)
        let current = &mut self.alpha_score;
        let state = &mut self.state;
        current.slice_mut(s![..l]).assign(&state.slice(s![..l]));
        // Compute the scores at (t, *)
        for t in 1..self.num_items as usize {
            // let (prev, current) = self.alpha_score.split_at_mut(l * t);
            let prev = self.alpha_score.clone();
            let prev = prev.slice(s![..l * t]);
            let prev: ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 1]>> = prev.slice(s![l * (t - 1)..]);

            let mut current = self.alpha_score.slice_mut(s![l * t ..]);

            let state = &self.state.slice(s![l * t..]);
            let mut back = self.backward_edge.slice_mut(s![l * t..]);
            // Compute the score of (t, j)
            for j in 0..l {
                let mut max_score = f64::MIN;
                let mut argmax_score = None;
                for (i, prev_value) in prev.iter().enumerate().take(l) {
                    // Transit from (t-1, i) to (t, j)
                    let trans = &self.trans.slice(s![l * i..]);
                    score = prev_value + trans[j];
                    // Store this path if it has the maximum score
                    if max_score < score {
                        max_score = score;
                        argmax_score = Some(i);
                    }
                }
                // Backward link (#t, #j) -> (#t-1, #i)
                if let Some(argmax_score) = argmax_score {
                    back[j] = argmax_score as u32;
                }
                // Add the state score on (t, j)
                current[j] = max_score + state[j];
            }
        }
        // Find the node (#T, Ei) that reaches EOS with the maximum score
        let mut max_score = f64::MIN;
        let prev = &self.alpha_score.slice(s![l * (self.num_items as usize - 1)..]);
        // Set a score for T-1 to be overwritten later. Just in case we don't
        // end up with something beating f64::MIN.
        let mut labels = vec![0u32; self.num_items as usize];
        for (i, prev_value) in prev.iter().enumerate().take(l) {
            if max_score < *prev_value {
                max_score = *prev_value;
                // Tag the item #T
                labels[self.num_items as usize - 1] = i as u32;
            }
        }
        // Tag labels by tracing teh backward links
        for t in (0..(self.num_items as usize - 1)).rev() {
            let back = &self.backward_edge.slice(s![l * (t + 1)..]);
            labels[t] = back[labels[t + 1] as usize];
        }
        (labels, max_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_new() {
        let _ctx = Context::new(Flag::VITERBI, 2, 0);
        let _ctx = Context::new(Flag::MARGINALS, 2, 0);
        let _ctx = Context::new(Flag::VITERBI | Flag::MARGINALS, 2, 0);
    }

    #[test]
    fn test_context_reset() {
        let mut ctx = Context::new(Flag::VITERBI | Flag::MARGINALS, 2, 0);
        ctx.reset(Reset::STATE);
        ctx.reset(Reset::TRANS);
        ctx.reset(Reset::STATE | Reset::TRANS);
    }
}
