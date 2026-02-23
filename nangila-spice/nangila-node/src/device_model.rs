//! Device Model Library
//!
//! Compact device models for nonlinear circuit elements.
//! Each model evaluates terminal currents and their derivatives
//! (conductances) for MNA stamping in the Newton-Raphson loop.
//!
//! Implemented models:
//!   - `DiodeModel` — Shockley diode equation (I = Is*(exp(V/Vt) - 1))
//!   - `MosfetLevel1` — Level-1 MOSFET (Shichman-Hodges, simplified BSIM4-style)
//!   - `ResistorModel` — Linear resistor (trivial, for testing)
//!
//! Phase 3, Sprint 9–10 deliverable.

// ─── Traits ────────────────────────────────────────────────────────

/// A linearised device stamp for MNA.
///
/// At each Newton iteration, a nonlinear device is approximated as:
///   I_device = G_eq * V + I_eq
///
/// G_eq (equivalent conductance) and I_eq (Norton current)
/// get stamped into the MNA G matrix and b vector.
#[derive(Debug, Clone)]
pub struct DeviceStamp {
    /// Equivalent conductance (∂I/∂V at current operating point)
    pub g_eq: f64,
    /// Norton equivalent current (I - G_eq * V)
    pub i_eq: f64,
    /// Positive terminal node (1-indexed, 0 = ground)
    pub node_p: usize,
    /// Negative terminal node (1-indexed, 0 = ground)
    pub node_n: usize,
}

/// Evaluation result from a device model at a given voltage.
#[derive(Debug, Clone)]
pub struct DeviceEval {
    /// Terminal current (amps, positive = into node_p)
    pub current: f64,
    /// ∂I/∂V (small-signal conductance)
    pub conductance: f64,
}

/// Trait for all compact device models.
pub trait DeviceModel {
    /// Evaluate the device at node voltages vp (node_p) and vn (node_n).
    fn eval(&self, vp: f64, vn: f64) -> DeviceEval;

    /// Generate the linearised MNA stamp for this operating point.
    fn stamp(&self, vp: f64, vn: f64) -> DeviceStamp;
}

// ─── Diode Model ───────────────────────────────────────────────────

/// Shockley ideal diode model.
///
/// I = Is * (exp(V/n*Vt) - 1)
/// where V = Vp - Vn, Vt = kT/q ≈ 25.85mV at 300K
#[derive(Debug, Clone)]
pub struct DiodeModel {
    /// Saturation current (A)
    pub is: f64,
    /// Thermal voltage (V) — kT/q = 25.85mV at 300K
    pub vt: f64,
    /// Ideality factor
    pub n: f64,
    /// node_p (anode)
    pub node_p: usize,
    /// node_n (cathode)
    pub node_n: usize,
}

impl DiodeModel {
    pub fn new(node_p: usize, node_n: usize) -> Self {
        Self {
            is: 1e-14,   // 10fA saturation current
            vt: 0.02585, // 25.85mV at 300K
            n: 1.0,
            node_p,
            node_n,
        }
    }

    pub fn with_params(mut self, is: f64, n: f64) -> Self {
        self.is = is;
        self.n = n;
        self
    }
}

impl DeviceModel for DiodeModel {
    fn eval(&self, vp: f64, vn: f64) -> DeviceEval {
        let v_d = vp - vn;
        let v_thermal = self.n * self.vt;

        // Limit exponential to avoid overflow (max junction voltage)
        let exp_arg = (v_d / v_thermal).min(40.0); // Limits to ~e^40 ≈ 2.4e17
        let exp_val = exp_arg.exp();

        let current = self.is * (exp_val - 1.0);
        let conductance = self.is * exp_val / v_thermal;

        DeviceEval {
            current,
            conductance,
        }
    }

    fn stamp(&self, vp: f64, vn: f64) -> DeviceStamp {
        let eval = self.eval(vp, vn);
        let v_d = vp - vn;

        // Norton equivalent: I_eq = I - G_eq * V
        let i_eq = eval.current - eval.conductance * v_d;

        DeviceStamp {
            g_eq: eval.conductance,
            i_eq,
            node_p: self.node_p,
            node_n: self.node_n,
        }
    }
}

// ─── Level-1 MOSFET Model ──────────────────────────────────────────

/// MOSFET operating region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MosfetRegion {
    Cutoff,
    Linear,
    Saturation,
}

/// Level-1 NMOS MOSFET (Shichman-Hodges model).
///
/// This is a simplified BSIM4-style model for Phase 3 bring-up.
/// It captures the essential nonlinear behavior:
///   - Cutoff:     Vgs < Vth → Id = 0
///   - Linear:     Vgs > Vth, Vds < Vgs-Vth → Id = β*((Vgs-Vth)*Vds - Vds²/2)
///   - Saturation: Vgs > Vth, Vds > Vgs-Vth → Id = β/2*(Vgs-Vth)²*(1+λ*Vds)
///
/// Where β = μn * Cox * W/L
#[derive(Debug, Clone)]
pub struct MosfetLevel1 {
    /// Threshold voltage (V)
    pub vth: f64,
    /// Process transconductance: μn * Cox * W/L (A/V²)
    pub beta: f64,
    /// Channel-length modulation (1/V)
    pub lambda: f64,
    /// Gate node (1-indexed)
    pub node_g: usize,
    /// Drain node (1-indexed)
    pub node_d: usize,
    /// Source node (1-indexed, 0 = ground)
    pub node_s: usize,
}

impl MosfetLevel1 {
    /// Standard NMOS with typical 180nm process parameters.
    pub fn nmos_180nm(node_g: usize, node_d: usize, node_s: usize) -> Self {
        Self {
            vth: 0.5,    // 500mV threshold
            beta: 1e-3,  // 1mA/V² (W/L=10, μnCox=100μA/V²)
            lambda: 0.1, // 0.1/V channel-length modulation
            node_g,
            node_d,
            node_s,
        }
    }

    /// Evaluate drain current and region.
    pub fn evaluate(&self, vgs: f64, vds: f64) -> (f64, f64, f64, MosfetRegion) {
        // Returns: (id, gm, gds, region)
        let vov = vgs - self.vth;

        if vov <= 0.0 {
            // Cutoff: Id = 0
            return (0.0, 0.0, 0.0, MosfetRegion::Cutoff);
        }

        if vds < vov {
            // Linear region
            let id = self.beta * ((vov * vds) - (vds * vds / 2.0));
            let gm = self.beta * vds; // ∂Id/∂Vgs
            let gds = self.beta * (vov - vds); // ∂Id/∂Vds
            (id, gm, gds, MosfetRegion::Linear)
        } else {
            // Saturation
            let id = (self.beta / 2.0) * vov * vov * (1.0 + self.lambda * vds);
            let gm = self.beta * vov * (1.0 + self.lambda * vds); // ∂Id/∂Vgs
            let gds = (self.beta / 2.0) * vov * vov * self.lambda; // ∂Id/∂Vds
            (id, gm, gds, MosfetRegion::Saturation)
        }
    }

    /// Generate linearised MNA stamps for the drain current.
    /// Returns: (gds_stamp, gm_stamp) for the drain-source conductance
    /// and transconductance controlled current source.
    pub fn mna_stamp(&self, voltages: &[f64]) -> (f64, f64, f64) {
        let vg = if self.node_g > 0 && self.node_g <= voltages.len() {
            voltages[self.node_g - 1]
        } else {
            0.0
        };
        let vd = if self.node_d > 0 && self.node_d <= voltages.len() {
            voltages[self.node_d - 1]
        } else {
            0.0
        };
        let vs = if self.node_s > 0 && self.node_s <= voltages.len() {
            voltages[self.node_s - 1]
        } else {
            0.0
        };

        let vgs = vg - vs;
        let vds = vd - vs;

        let (id, gm, gds, _region) = self.evaluate(vgs, vds);

        // Norton equivalent: I_drain_eq = Id - gm*Vgs - gds*Vds
        let i_eq = id - gm * vgs - gds * vds;

        (gm, gds, i_eq)
    }

    /// Get the current operating region.
    pub fn region(&self, voltages: &[f64]) -> MosfetRegion {
        let vg = if self.node_g > 0 {
            voltages.get(self.node_g - 1).copied().unwrap_or(0.0)
        } else {
            0.0
        };
        let vd = if self.node_d > 0 {
            voltages.get(self.node_d - 1).copied().unwrap_or(0.0)
        } else {
            0.0
        };
        let vs = if self.node_s > 0 {
            voltages.get(self.node_s - 1).copied().unwrap_or(0.0)
        } else {
            0.0
        };

        let (_, _, _, region) = self.evaluate(vg - vs, vd - vs);
        region
    }
}

// ─── Linear Resistor (for testing/baseline) ────────────────────────

/// Linear resistor — trivial device model.
#[derive(Debug, Clone)]
pub struct ResistorModel {
    pub resistance: f64,
    pub node_p: usize,
    pub node_n: usize,
}

impl DeviceModel for ResistorModel {
    fn eval(&self, vp: f64, vn: f64) -> DeviceEval {
        let g = 1.0 / self.resistance;
        let v = vp - vn;
        DeviceEval {
            current: g * v,
            conductance: g,
        }
    }

    fn stamp(&self, vp: f64, vn: f64) -> DeviceStamp {
        let g = 1.0 / self.resistance;
        DeviceStamp {
            g_eq: g,
            i_eq: 0.0, // Linear: no Norton current offset
            node_p: self.node_p,
            node_n: self.node_n,
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diode_forward_bias() {
        let diode = DiodeModel::new(1, 0);

        // At Vd = 0.7V (silicon forward bias)
        let eval = diode.eval(0.7, 0.0);
        assert!(eval.current > 0.0, "Forward biased diode should conduct");
        assert!(eval.conductance > 0.0, "Conductance should be positive");

        // Exponential growth: I at 0.7V should be much larger than at 0.1V
        let eval_low = diode.eval(0.1, 0.0);
        assert!(
            eval.current > eval_low.current * 100.0,
            "Diode current should grow exponentially"
        );
    }

    #[test]
    fn test_diode_reverse_bias() {
        let diode = DiodeModel::new(1, 0);
        let eval = diode.eval(-1.0, 0.0); // Reverse biased

        // Should approach -Is (saturation current)
        assert!(
            eval.current < 0.0,
            "Reverse bias: current should be negative"
        );
        assert!(
            eval.current.abs() < diode.is * 2.0,
            "Reverse current bounded by Is"
        );
    }

    #[test]
    fn test_diode_stamp() {
        let diode = DiodeModel::new(1, 0);
        let stamp = diode.stamp(0.6, 0.0);

        assert!(stamp.g_eq > 0.0, "G_eq should be positive");
        // Norton: I_eq = I - G_eq * V
        let v = 0.6;
        let eval = diode.eval(v, 0.0);
        let expected_i_eq = eval.current - eval.conductance * v;
        assert!(
            (stamp.i_eq - expected_i_eq).abs() < 1e-12,
            "Norton current I_eq mismatch"
        );
    }

    #[test]
    fn test_mosfet_regions() {
        let fet = MosfetLevel1::nmos_180nm(1, 2, 0);

        // Cutoff: Vgs < Vth (0.5V)
        let (id, _, _, region) = fet.evaluate(0.2, 1.0);
        assert_eq!(region, MosfetRegion::Cutoff);
        assert_eq!(id, 0.0);

        // Saturation: Vgs = 1V, Vds = 1.5V > Vov = 0.5V
        let (id_sat, gm, gds, region) = fet.evaluate(1.0, 1.5);
        assert_eq!(region, MosfetRegion::Saturation);
        assert!(id_sat > 0.0);
        assert!(gm > 0.0);
        assert!(gds >= 0.0);

        // Linear: Vgs = 1V, Vds = 0.2V < Vov = 0.5V
        let (id_lin, _, _, region) = fet.evaluate(1.0, 0.2);
        assert_eq!(region, MosfetRegion::Linear);
        assert!(id_lin > 0.0);
        assert!(
            id_lin < id_sat,
            "Linear region Id should be less than saturation for small Vds"
        );
    }

    #[test]
    fn test_mosfet_channel_length_modulation() {
        let fet = MosfetLevel1::nmos_180nm(1, 2, 0);

        // In saturation, higher Vds → higher Id due to CLM
        let (id1, _, _, _) = fet.evaluate(1.0, 1.5);
        let (id2, _, _, _) = fet.evaluate(1.0, 3.0);

        assert!(id2 > id1, "CLM: Id should increase with Vds in saturation");
    }

    #[test]
    fn test_resistor_model() {
        let r = ResistorModel {
            resistance: 1000.0,
            node_p: 1,
            node_n: 0,
        };

        let eval = r.eval(1.0, 0.0);
        assert!(
            (eval.current - 1e-3).abs() < 1e-10,
            "1V / 1kΩ = 1mA, got {:.4e}",
            eval.current
        );
        assert!(
            (eval.conductance - 1e-3).abs() < 1e-10,
            "Conductance = 1/R = 1mS"
        );
    }

    #[test]
    fn test_mosfet_mna_stamp() {
        let fet = MosfetLevel1::nmos_180nm(1, 2, 0);
        // Vg=1.2V (node 1), Vd=1.8V (node 2), Vs=0 (ground)
        let voltages = vec![1.2, 1.8];
        let (gm, gds, i_eq) = fet.mna_stamp(&voltages);

        assert!(gm >= 0.0, "Transconductance should be non-negative");
        assert!(gds >= 0.0, "Output conductance should be non-negative");
        // i_eq compensates for the linearisation offset
        // Just check it's finite
        assert!(i_eq.is_finite(), "Norton current should be finite");
    }
}
