param(
    [string]$OutputDir = "D:\datasets\zod-bev-v2-private"
)

$ErrorActionPreference = "Stop"
$repository = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$output = [System.IO.Path]::GetFullPath($OutputDir)
if ($output.StartsWith($repository, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "The private dataset output must remain outside the repository."
}

$locator = Get-Clipboard -Raw
if (-not $locator -or $locator -notmatch '^https://www\.dropbox\.com/scl/fo/') {
    throw "Copy the granted ZOD Dropbox folder URL, then run this script again."
}

New-Item -ItemType Directory -Path $output -Force | Out-Null
$python = Join-Path $repository ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "The project virtual environment is unavailable: $python"
}

$worker = Join-Path $output "start_frames_metadata_worker.ps1"
$stdout = Join-Path $output "frames_metadata_download.stdout.log"
$stderr = Join-Path $output "frames_metadata_download.stderr.log"
$workerBody = @'
param([string]$Python, [string]$Repository, [string]$Output)
try {
    & $Python (Join-Path $Repository "scripts\download_zod_subset.py") `
        --subset frames `
        --archive annotations.tar.gz `
        --archive infos.tar.gz `
        --output-dir $Output `
        --execute `
        --remove-archives
    exit $LASTEXITCODE
}
finally {
    Remove-Item Env:\ZOD_DROPBOX_URL -ErrorAction SilentlyContinue
}
'@
Set-Content -LiteralPath $worker -Value $workerBody -Encoding UTF8
$env:ZOD_DROPBOX_URL = $locator
try {
    $process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", ('"' + $worker + '"'),
            "-Python", ('"' + $python + '"'),
            "-Repository", ('"' + $repository + '"'),
            "-Output", ('"' + $output + '"')
        ) `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
}
finally {
    Remove-Item Env:\ZOD_DROPBOX_URL -ErrorAction SilentlyContinue
}

try {
    Set-Clipboard -Value ""
}
catch {
    Write-Warning "The clipboard could not be cleared automatically."
}
$locator = $null
Write-Host "Secure Frames metadata worker started (PID $($process.Id))."
Write-Host "Private logs: $stdout and $stderr"
