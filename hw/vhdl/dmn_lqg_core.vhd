-- =============================================================================
--  dmn_lqg_core : the M1 closed loop (stationary Kalman filter + LQG regulator)
--  as a three-layer neural circuit, one epoch (10 s) per `start` pulse.
--
--      layer 1  PREDICT  x_pred = F xhat + B u + c            2 neurons, 3 inputs
--      layer 2  CORRECT  xhat   = (I-L) x_pred + L y          2 neurons, 4 inputs
--      layer 3  CONTROL  u      = clip(-K xhat + b3, +-u_max) 1 neuron
--               READOUT  qoc    = h xhat + gamma0             1 neuron (monitor)
--
--  Timing: 3 clock cycles per epoch (one layer per cycle), `done` is high for
--  one cycle when u / xhat / qoc_hat are valid.  On reset the state is the
--  stationary mean MU_Q and u = control(MU_Q), exactly as NeuralCircuit.reset()
--  in model/hardware.py.  All constants come from gen/lqg_params_pkg.vhd.
-- =============================================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.lqg_params_pkg.all;
use work.neuron_pkg.all;

entity dmn_lqg_core is
  port (
    clk      : in  std_logic;
    rst      : in  std_logic;                     -- synchronous, active high
    start    : in  std_logic;                     -- one pulse per epoch with new y
    y_a      : in  signed(W - 1 downto 0);        -- EEG feature for A_DMN (Q format)
    y_phi    : in  signed(W - 1 downto 0);        -- EEG feature for Phi
    u_out    : out signed(W - 1 downto 0);        -- control input for the next epoch
    xhat_a   : out signed(W - 1 downto 0);
    xhat_phi : out signed(W - 1 downto 0);
    qoc_hat  : out signed(W - 1 downto 0);        -- estimated QoC (monitoring)
    done     : out std_logic
  );
end entity dmn_lqg_core;

architecture rtl of dmn_lqg_core is
  type state_t is (IDLE, PREDICT, CORRECT, CONTROL);
  signal state   : state_t := IDLE;
  signal xhat    : fx_vec(0 to 1) := (to_signed(MU_Q(0), W), to_signed(MU_Q(1), W));
  signal x_pred  : fx_vec(0 to 1) := (others => (others => '0'));
  signal u_r     : fx_t := (others => '0');
  signal qoc_r   : fx_t := (others => '0');
  signal y_r     : fx_vec(0 to 1) := (others => (others => '0'));
  signal done_r  : std_logic := '0';

  constant MU_FX : fx_vec(0 to 1) := (to_signed(MU_Q(0), W), to_signed(MU_Q(1), W));
begin

  process (clk)
    variable in1 : fx_vec(0 to 2);
    variable in2 : fx_vec(0 to 3);
  begin
    if rising_edge(clk) then
      done_r <= '0';
      if rst = '1' then
        state  <= IDLE;
        xhat   <= MU_FX;
        u_r    <= clip(neuron(MU_FX, W3, B3), UMAX_Q);
        qoc_r  <= neuron(MU_FX, W4, B4);
      else
        case state is
          when IDLE =>
            if start = '1' then
              y_r(0) <= y_a;
              y_r(1) <= y_phi;
              state  <= PREDICT;
            end if;

          when PREDICT =>                        -- layer 1
            in1 := (xhat(0), xhat(1), u_r);
            x_pred(0) <= neuron(in1, W1_A, B1_A);
            x_pred(1) <= neuron(in1, W1_PHI, B1_PHI);
            state <= CORRECT;

          when CORRECT =>                        -- layer 2
            in2 := (x_pred(0), x_pred(1), y_r(0), y_r(1));
            xhat(0) <= neuron(in2, W2_A, B2_A);
            xhat(1) <= neuron(in2, W2_PHI, B2_PHI);
            state <= CONTROL;

          when CONTROL =>                        -- layer 3 + read-out
            u_r    <= clip(neuron(xhat, W3, B3), UMAX_Q);
            qoc_r  <= neuron(xhat, W4, B4);
            done_r <= '1';
            state  <= IDLE;
        end case;
      end if;
    end if;
  end process;

  u_out    <= u_r;
  xhat_a   <= xhat(0);
  xhat_phi <= xhat(1);
  qoc_hat  <= qoc_r;
  done     <= done_r;

end architecture rtl;
