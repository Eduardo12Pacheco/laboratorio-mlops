#requires -Version 5.1
<#
.SYNOPSIS
    Installer for the mlops-lifecycle-lab lab (L2 Docker Compose and L1 uv path).

.DESCRIPTION
    Runs the prerequisite checks, prepares a local .env file and starts the Compose
    stack, or installs the CLI with uv for the Docker-free path. Windows PowerShell 5.1
    and PowerShell 7 are both supported.

.PARAMETER Check
    Run the prerequisite checks only (no side effects).

.PARAMETER L1
    Use the Docker-free L1 path (uv, Python 3.12 and `mll doctor`).

.PARAMETER Uninstall
    Stop the stack and remove its volumes and local images.

.PARAMETER Yes
    Do not ask for confirmation on -Uninstall.

.PARAMETER NoStart
    Prepare the environment but do not build or start the stack.

.PARAMETER ProjectName
    Compose project name (default: mlops-lifecycle-lab).

.NOTES
    Exit codes: 0 ok, 1 prerequisites failed, 2 usage error, 3 install failed,
    4 uninstall aborted.
#>
[CmdletBinding()]
param(
    [switch]$Check,
    [switch]$L1,
    [switch]$Uninstall,
    [switch]$Yes,
    [switch]$NoStart,
    [string]$ProjectName = "mlops-lifecycle-lab"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Ports = @(14000, 14080, 14501)
$MinimumFreeBytes = 4GB
$script:Failures = 0
$LASTEXITCODE = 0

function Write-Ok {
    param([string]$Message)
    Write-Host "[ ok ] $Message"
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[warn] $Message"
}

function Write-Fail {
    param([string]$Message)
    Write-Host "[FAIL] $Message"
    $script:Failures = $script:Failures + 1
}

function Show-Usage {
    Write-Host @"
Usage: .\install.ps1 [options]

Options:
  -Check              Run the prerequisite checks only (no side effects).
  -L1                 Use the Docker-free L1 path (uv, Python 3.12, mll doctor).
  -Uninstall          Stop the stack and remove its volumes and local images.
  -Yes                Do not ask for confirmation on -Uninstall.
  -NoStart            Prepare the environment but do not build or start the stack.
  -ProjectName NAME   Compose project name (default: mlops-lifecycle-lab).

Exit codes: 0 ok, 1 prerequisites failed, 2 usage error, 3 install failed,
4 uninstall aborted.
"@
}

function Invoke-Compose {
    param([string[]]$Arguments)
    & docker compose -p $ProjectName @Arguments
    return $LASTEXITCODE
}

function Test-ComposeProjectRunning {
    $containers = & docker compose -p $ProjectName ps --status running -q 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    foreach ($container in $containers) {
        if (-not [string]::IsNullOrWhiteSpace($container)) {
            return $true
        }
    }
    return $false
}

function Test-PortInUse {
    param([int]$Port)
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync('127.0.0.1', $Port)
        if (-not $task.Wait(500)) {
            return $false
        }
        return $client.Connected
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Test-Docker {
    if ($null -eq (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Fail "docker not found (install Docker >= 24 with the Compose v2 plugin)"
        return
    }
    Write-Ok "docker found"

    $dockerVersion = (& docker --version 2>$null) -join ' '
    if ($dockerVersion -match '([0-9]+)\.[0-9]+') {
        $major = [int]$Matches[1]
        if ($major -ge 24) {
            Write-Ok "docker >= 24 ($dockerVersion)"
        } else {
            Write-Fail "docker >= 24 is required (found: $dockerVersion)"
        }
    } else {
        Write-Fail "could not determine the docker version"
    }

    $composeVersion = (& docker compose version 2>$null) -join ' '
    if ($composeVersion -match 'v?([0-9]+)\.[0-9]+' -and [int]$Matches[1] -ge 2) {
        Write-Ok "compose v2 ($composeVersion)"
    } else {
        Write-Fail "docker compose v2 is required (found: $composeVersion)"
    }

    & docker info *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "docker daemon reachable"
    } else {
        Write-Fail "docker daemon is not reachable"
    }

    $engineOs = (& docker info --format '{{.OSType}}' 2>$null) -join ''
    $engineOs = $engineOs.Trim().ToLowerInvariant()
    if ($engineOs -eq 'linux') {
        Write-Ok "docker engine runs Linux containers"
    } elseif ([string]::IsNullOrWhiteSpace($engineOs)) {
        Write-Warn "could not determine the docker engine OS"
    } else {
        Write-Warn "Docker engine OS is '$engineOs'; this stack needs Linux containers"
    }
}

function Test-Ports {
    if (Test-ComposeProjectRunning) {
        Write-Ok "host ports are owned by compose project '$ProjectName'"
        return
    }
    foreach ($port in $Ports) {
        if (Test-PortInUse -Port $port) {
            Write-Fail "port $port is already in use (stop the process or change -ProjectName)"
        } else {
            Write-Ok "port $port is free"
        }
    }
}

function Test-Disk {
    try {
        $drive = New-Object System.IO.DriveInfo($RepoRoot)
        $free = $drive.AvailableFreeSpace
        if ($free -lt $MinimumFreeBytes) {
            Write-Fail ("at least 4 GB of free disk is required (found: {0} MB)" -f [int]($free / 1MB))
        } else {
            Write-Ok ("at least 4 GB free disk ({0} MB)" -f [int]($free / 1MB))
        }
    } catch {
        Write-Warn "could not determine free disk space"
    }
}

function Test-EnvFile {
    if (Test-Path (Join-Path $RepoRoot '.env')) {
        Write-Ok ".env present"
    } else {
        Write-Warn ".env not found (install creates it from .env.example)"
    }
}

function Test-L1 {
    if ($null -eq (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Fail "uv not found (install it from https://docs.astral.sh/uv/)"
        return
    }
    Push-Location $RepoRoot
    try {
        $uvVersion = (& uv --version 2>$null) -join ' '
        Write-Ok "uv found ($uvVersion)"

        & uv python find 3.12 *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Python 3.12 resolvable with uv"
        } else {
            Write-Fail "Python 3.12 is not resolvable with uv (run: uv python install 3.12)"
        }

        & uv run mll doctor *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "mll doctor"
        } else {
            Write-Fail "mll doctor failed (run: uv run mll doctor)"
        }
    } finally {
        Pop-Location
    }
}

function Test-L2 {
    Test-Docker
    Test-Ports
    Test-Disk
    Test-EnvFile
}

function Invoke-Checks {
    $script:Failures = 0
    if ($L1) {
        Test-L1
    } else {
        Test-L2
    }
    return ($script:Failures -eq 0)
}

function New-EnvFile {
    $envPath = Join-Path $RepoRoot '.env'
    if (Test-Path $envPath) {
        Write-Ok ".env already exists"
    } else {
        Copy-Item (Join-Path $RepoRoot '.env.example') $envPath
        Write-Ok "created .env from .env.example"
    }
}

function Show-Urls {
    Write-Host ""
    Write-Host "Control Room: http://127.0.0.1:14501"
    Write-Host "API docs:     http://127.0.0.1:14080/docs"
    Write-Host "MLflow:       http://127.0.0.1:14000"
}

function Invoke-Uninstall {
    if ($null -eq (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Fail "docker not found"
        exit 1
    }
    if (-not $Yes) {
        $answer = Read-Host "Remove compose project '$ProjectName' with its volumes and local images? [y/N]"
        if ($answer -notmatch '^(y|yes)$') {
            Write-Host "uninstall aborted"
            exit 4
        }
    }
    $code = Invoke-Compose -Arguments @('down', '-v', '--rmi', 'local')
    if ($code -ne 0) {
        Write-Fail "docker compose down failed"
        exit 3
    }
    Write-Ok "uninstalled compose project '$ProjectName' (.env kept)"
    exit 0
}

function Invoke-InstallL1 {
    if (-not (Invoke-Checks)) {
        Write-Host ""
        Write-Host "L1 prerequisites failed"
        exit 1
    }
    Push-Location $RepoRoot
    try {
        & uv tool install --force $RepoRoot
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "uv tool install failed"
            exit 3
        }
    } finally {
        Pop-Location
    }
    Write-Ok "installed the mll CLI with uv"
    Write-Host ""
    Write-Host "Next: mll quickstart"
    Write-Host "      mll dashboard"
    exit 0
}

function Invoke-InstallL2 {
    $alreadyInstalled = $false

    if (-not (Invoke-Checks)) {
        Write-Host ""
        Write-Host "prerequisite checks failed; nothing was started."
        exit 1
    }

    New-EnvFile

    if ($NoStart) {
        Write-Ok "-NoStart requested; the stack was not started"
        Show-Urls
        exit 0
    }

    if (Test-ComposeProjectRunning) {
        $alreadyInstalled = $true
    }

    $code = Invoke-Compose -Arguments @('up', '-d', '--build', '--wait')
    if ($code -ne 0) {
        Write-Fail "docker compose up failed"
        exit 3
    }

    if ($alreadyInstalled) {
        Write-Ok "already installed, updated"
    } else {
        Write-Ok "installed compose project '$ProjectName'"
    }
    Show-Urls
    exit 0
}

if ($ProjectName -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]*$') {
    Write-Host "error: invalid compose project name: $ProjectName"
    Show-Usage
    exit 2
}

if ($Uninstall -and ($Check -or $L1 -or $NoStart)) {
    Write-Host "error: -Uninstall cannot be combined with -Check, -L1 or -NoStart"
    Show-Usage
    exit 2
}

if ($Uninstall) {
    Invoke-Uninstall
}

if ($Check) {
    if (Invoke-Checks) {
        Write-Host ""
        Write-Host "Prerequisites OK"
        exit 0
    }
    Write-Host ""
    Write-Host "Prerequisites FAILED"
    exit 1
}

if ($L1) {
    Invoke-InstallL1
}

Invoke-InstallL2
Show-Usage
exit 2
