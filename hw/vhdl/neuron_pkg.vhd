-- =============================================================================
--  neuron_pkg : linear neuron with rounding and saturation (fixed point)
--
--  Reference semantics = model/hardware.py :: neuron()
--      acc = (bias << FRAC) + sum_i w_i * x_i          exact, ACC_W bits
--      y   = saturate_W( (acc + 2^(FRAC-1)) >>a FRAC )  round half up, arithmetic shift
--  W and FRAC come from the generated package lqg_params_pkg.
-- =============================================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.lqg_params_pkg.all;

package neuron_pkg is
  constant ACC_W : natural := 2 * W + 3;
  subtype fx_t is signed(W - 1 downto 0);
  type fx_vec is array (natural range <>) of fx_t;

  function saturate(a : signed) return fx_t;
  function neuron(x : fx_vec; wt : int_vec; bias : integer) return fx_t;
  function clip(x : fx_t; lim : integer) return fx_t;
end package neuron_pkg;

package body neuron_pkg is

  function saturate(a : signed) return fx_t is
    constant HI : signed(a'length - 1 downto 0) := to_signed(2 ** (W - 1) - 1, a'length);
    constant LO : signed(a'length - 1 downto 0) := to_signed(-(2 ** (W - 1)), a'length);
  begin
    if a > HI then
      return resize(HI, W);
    elsif a < LO then
      return resize(LO, W);
    else
      return resize(a, W);
    end if;
  end function;

  -- NB: VHDL identifiers are case-insensitive, so the weight parameter must not
  -- be called "w" (it would hide the word-width constant W).
  function neuron(x : fx_vec; wt : int_vec; bias : integer) return fx_t is
    variable acc : signed(ACC_W - 1 downto 0);
  begin
    assert x'length = wt'length report "neuron: input/weight length mismatch" severity failure;
    acc := shift_left(to_signed(bias, ACC_W), FRAC);
    for i in 0 to x'length - 1 loop
      acc := acc + resize(x(x'low + i) * to_signed(wt(wt'low + i), W), ACC_W);
    end loop;
    acc := acc + to_signed(2 ** (FRAC - 1), ACC_W);
    acc := shift_right(acc, FRAC);          -- arithmetic shift on signed
    return saturate(acc);
  end function;

  function clip(x : fx_t; lim : integer) return fx_t is
  begin
    if x > to_signed(lim, W) then
      return to_signed(lim, W);
    elsif x < to_signed(-lim, W) then
      return to_signed(-lim, W);
    else
      return x;
    end if;
  end function;

end package body neuron_pkg;
