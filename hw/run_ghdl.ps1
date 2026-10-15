# Bit-exact simulation of dmn_lqg_core with GHDL (Windows PowerShell).
# Needs GHDL (https://github.com/ghdl/ghdl/releases, "Windows (x86-64, UCRT64, standalone) - mcode backend")
# and the generated files from:  python main.py model hardware
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File hw\run_ghdl.ps1
# Waveform for GTKWave / Surfer:  add  -Wave   -> hw\vhdl\work\tb_dmn_lqg_core.ghw (+ .vcd)
param([switch]$Wave)
$ErrorActionPreference = "Stop"

# Locate ghdl.exe: (1) current PATH, (2) PATH re-read from the registry (terminal opened
# before PATH was changed), (3) $env:GHDL, (4) C:\ghdl*\bin\ghdl.exe
function Find-Ghdl {
    $cmd = Get-Command ghdl -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
    $cmd = Get-Command ghdl -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    if ($env:GHDL -and (Test-Path $env:GHDL)) { return $env:GHDL }
    $hit = Get-ChildItem -Path "C:\" -Directory -Filter "ghdl*" -ErrorAction SilentlyContinue |
           ForEach-Object { Join-Path $_.FullName "bin\ghdl.exe" } |
           Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($hit) { return $hit }
    throw "GHDL not found. Install it and add its bin folder to PATH (see hw/README.md)."
}

$ghdl = Find-Ghdl
Write-Host "Using $ghdl"
Set-Location (Join-Path $PSScriptRoot "vhdl")
if (-not (Test-Path "gen\lqg_params_pkg.vhd")) { throw "gen\lqg_params_pkg.vhd missing: run  python main.py model hardware  first" }
New-Item -ItemType Directory -Force work | Out-Null

& $ghdl -a --std=08 --workdir=work gen\lqg_params_pkg.vhd neuron_pkg.vhd dmn_lqg_core.vhd tb_dmn_lqg_core.vhd
if ($LASTEXITCODE -ne 0) { throw "ghdl analysis failed" }
& $ghdl -e --std=08 --workdir=work tb_dmn_lqg_core
if ($LASTEXITCODE -ne 0) { throw "ghdl elaboration failed" }
# the -g generic must be quoted, otherwise PowerShell splits it at the dot
$runArgs = @("-r", "--std=08", "--workdir=work", "tb_dmn_lqg_core", "-gVEC_FILE=gen/vectors.txt", "--assert-level=error")
if ($Wave) { $runArgs += @("--wave=work/tb_dmn_lqg_core.ghw", "--vcd=work/tb_dmn_lqg_core.vcd") }
& $ghdl @runArgs
if ($LASTEXITCODE -ne 0) { throw "simulation FAILED (see MISMATCH lines above)" }
if ($Wave) {
    Write-Host "Waveforms: $(Resolve-Path work\tb_dmn_lqg_core.ghw)"
    Write-Host "           $(Resolve-Path work\tb_dmn_lqg_core.vcd)"
}
