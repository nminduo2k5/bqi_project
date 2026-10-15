-- =============================================================================
--  tb_dmn_lqg_core : bit-exact check of dmn_lqg_core against gen/vectors.txt
--  (produced by python main.py model hardware).
--
--  vectors.txt format:
--      # comment lines
--      N
--      y_a y_phi xhat_a xhat_phi u qoc     (N lines, integers in Q format)
--  Each line is one epoch: the testbench applies (y_a, y_phi), pulses `start`,
--  waits for `done` and compares the four outputs with the expected integers.
--  Run:  ghdl -a --std=08 gen/lqg_params_pkg.vhd neuron_pkg.vhd dmn_lqg_core.vhd tb_dmn_lqg_core.vhd
--        ghdl -e --std=08 tb_dmn_lqg_core
--        ghdl -r --std=08 tb_dmn_lqg_core -gVEC_FILE=gen/vectors.txt
-- =============================================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use work.lqg_params_pkg.all;
use work.neuron_pkg.all;

entity tb_dmn_lqg_core is
  generic (VEC_FILE : string := "gen/vectors.txt");
end entity tb_dmn_lqg_core;

architecture sim of tb_dmn_lqg_core is
  constant T_CLK : time := 10 ns;
  signal clk      : std_logic := '0';
  signal rst      : std_logic := '1';
  signal start    : std_logic := '0';
  signal y_a      : signed(W - 1 downto 0) := (others => '0');
  signal y_phi    : signed(W - 1 downto 0) := (others => '0');
  signal u_out    : signed(W - 1 downto 0);
  signal xhat_a   : signed(W - 1 downto 0);
  signal xhat_phi : signed(W - 1 downto 0);
  signal qoc_hat  : signed(W - 1 downto 0);
  signal done     : std_logic;
  signal finished : boolean := false;
begin

  dut : entity work.dmn_lqg_core
    port map (clk => clk, rst => rst, start => start, y_a => y_a, y_phi => y_phi,
              u_out => u_out, xhat_a => xhat_a, xhat_phi => xhat_phi, qoc_hat => qoc_hat, done => done);

  clk <= not clk after T_CLK / 2 when not finished else '0';

  stimulus : process
    file vec       : text;
    variable l     : line;
    variable ch    : character;
    variable n     : integer;
    variable v_ya, v_yp, e_xa, v_dummy : integer;
    variable e_xp, e_u, e_q : integer;
    variable errors : integer := 0;
    variable k     : integer := 0;
    variable good  : boolean;
  begin
    file_open(vec, VEC_FILE, read_mode);
    -- skip comment lines, then read N
    loop
      readline(vec, l);
      if l'length > 0 and l(1) /= '#' then
        read(l, n);
        exit;
      end if;
    end loop;
    report "tb_dmn_lqg_core: " & integer'image(n) & " vectors, W=" & integer'image(W) &
           " FRAC=" & integer'image(FRAC);

    -- reset
    rst <= '1';
    wait for 3 * T_CLK;
    wait until rising_edge(clk);
    rst <= '0';
    wait until rising_edge(clk);

    for i in 1 to n loop
      readline(vec, l);
      read(l, v_ya);
      read(l, v_yp);
      read(l, e_xa);
      read(l, e_xp);
      read(l, e_u);
      read(l, e_q);

      y_a   <= to_signed(v_ya, W);
      y_phi <= to_signed(v_yp, W);
      start <= '1';
      wait until rising_edge(clk);
      start <= '0';
      -- wait for done (3 cycles)
      loop
        wait until rising_edge(clk);
        exit when done = '1';
      end loop;
      wait for 1 ns;   -- let outputs settle after the edge
      good := (to_integer(xhat_a) = e_xa) and (to_integer(xhat_phi) = e_xp) and
              (to_integer(u_out) = e_u) and (to_integer(qoc_hat) = e_q);
      if not good then
        errors := errors + 1;
        if errors <= 10 then
          report "MISMATCH epoch " & integer'image(i) &
                 ": xhat=(" & integer'image(to_integer(xhat_a)) & "," & integer'image(to_integer(xhat_phi)) &
                 ") u=" & integer'image(to_integer(u_out)) & " qoc=" & integer'image(to_integer(qoc_hat)) &
                 "  expected (" & integer'image(e_xa) & "," & integer'image(e_xp) & ") u=" &
                 integer'image(e_u) & " qoc=" & integer'image(e_q)
            severity warning;
        end if;
      end if;
      k := k + 1;
    end loop;
    file_close(vec);

    if errors = 0 then
      report "PASS: " & integer'image(k) & " epochs bit-exact with model/hardware.py" severity note;
    else
      report "FAIL: " & integer'image(errors) & " of " & integer'image(k) & " epochs differ" severity error;
    end if;
    finished <= true;
    wait;
  end process;

end architecture sim;
