$ErrorActionPreference = 'Stop'

Set-Location (Split-Path -Parent $PSCommandPath) | Out-Null
$RepoRoot = Resolve-Path (Join-Path (Get-Location) "..\\..") | Select-Object -ExpandProperty Path
Set-Location $RepoRoot | Out-Null

Write-Host "== Device profile (Windows 11) ==" -ForegroundColor Cyan
Write-Host "Repo: $RepoRoot"
Write-Host ""

Write-Host "## OS / CPU / RAM" -ForegroundColor Yellow
try {
  $os = Get-CimInstance Win32_OperatingSystem
  $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
  $ramGiB = [Math]::Round($os.TotalVisibleMemorySize / 1024 / 1024, 2)
  Write-Host ("OS: {0} {1}" -f $os.Caption, $os.Version)
  Write-Host ("CPU: {0}" -f $cpu.Name)
  Write-Host ("RAM (GiB): {0}" -f $ramGiB)
} catch {
  Write-Host "Could not query OS/CPU via CIM: $_"
}

Write-Host ""
Write-Host "## GPU" -ForegroundColor Yellow
try {
  Get-CimInstance Win32_VideoController |
    Select-Object Name, AdapterRAM, DriverVersion |
    ForEach-Object {
      $ram = if ($_.AdapterRAM) { [Math]::Round($_.AdapterRAM / 1GB, 2) } else { $null }
      Write-Host ("GPU: {0} (VRAM GiB: {1}) Driver: {2}" -f $_.Name, $ram, $_.DriverVersion)
    }
} catch {
  Write-Host "Could not query GPU via CIM: $_"
}

Write-Host ""
Write-Host "## Python env" -ForegroundColor Yellow
python -c "import sys; print('python:', sys.version)"
python -c "import cv2, numpy; print('cv2:', cv2.__version__); print('numpy:', numpy.__version__)" 2>$null
python -c "import ultralytics; print('ultralytics: ok')" 2>$null
python -c "import supervision; print('supervision: ok')" 2>$null
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())" 2>$null
python -c "import transformers; print('transformers:', transformers.__version__)" 2>$null

Write-Host ""
Write-Host "## Pipeline device profile (video-only, includes VLM if enabled)" -ForegroundColor Yellow
Write-Host "(Note: output includes layer-by-layer ROI/YOLO/VLM summary)"
python "src/evaluation-output-layer/benchmark.py"

