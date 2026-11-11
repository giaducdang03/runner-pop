# Tự chọn backend theo GPU của máy rồi chạy docker compose: NVIDIA -> cuda, Intel -> openvino, còn lại -> cpu.
# Dùng:  .\run.ps1            (chạy nền, build nếu cần)
#        .\run.ps1 down       (dừng)   |   .\run.ps1 -Mode cpu   (ép chế độ)
param([ValidateSet("auto", "cpu", "intel", "cuda")][string]$Mode = "auto", [Parameter(ValueFromRemainingArguments)]$Rest)

if ($Mode -eq "auto") {
    $names = (Get-CimInstance Win32_VideoController).Name -join "|"
    if ($names -match "NVIDIA") { $Mode = "cuda" }
    elseif ($names -match "Intel.*(Arc|Iris|UHD|Xe)") { $Mode = "intel" }
    else { $Mode = "cpu" }
}
Write-Host "Backend: $Mode"

$files = @("-f", "docker-compose.yml")
if ($Mode -ne "cpu") { $files += @("-f", "docker-compose.$Mode.yml") }
if (-not $Rest) { $Rest = @("up", "-d", "--build") }
docker compose @files @Rest
