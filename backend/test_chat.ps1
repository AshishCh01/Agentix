param (
    [string]$Message = "Hello! Who are you and what can you do?"
)

$supabaseUrl = "https://lufjqvomideqwowgrpid.supabase.co"
$anonKey     = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx1Zmpxdm9taWRlcXdvd2dycGlkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODYwMzIyMDAsImV4cCI6MjEwMTYwODIwMH0.FdM_pUJ3x1az4lnDvh24fgvqYhXgrTkuu0t9hdsMbVA"
$email       = "testuser@gmail.com"
$password    = "Solana2.0"
$baseUrl     = "http://127.0.0.1:8000/api/v1"

Write-Host "1. Fetching fresh access token..." -ForegroundColor Cyan
$authHeaders = @{ "apikey" = $anonKey; "Content-Type" = "application/json" }
$authBody    = @{ email = $email; password = $password } | ConvertTo-Json
$authRes     = Invoke-RestMethod -Uri "$supabaseUrl/auth/v1/token?grant_type=password" -Method Post -Headers $authHeaders -Body $authBody
$token       = $authRes.access_token

$apiHeaders  = @{ "Authorization" = "Bearer $token"; "Content-Type" = "application/json" }

$sessionFile = ".dev_session_id"
if (Test-Path $sessionFile) {
    # Extract string content cleanly without PowerShell object metadata
    $sessionId = (Get-Content -Path $sessionFile -Raw).Trim()
    Write-Host "2. Using existing Session ID: $sessionId" -ForegroundColor Yellow
} else {
    Write-Host "2. Creating new Chat Session..." -ForegroundColor Cyan
    $sessionBody = @{ title = "Automated Dev Session" } | ConvertTo-Json
    $sessionRes  = Invoke-RestMethod -Uri "$baseUrl/sessions" -Method Post -Headers $apiHeaders -Body $sessionBody
    $sessionId   = [string]$sessionRes.id
    [System.IO.File]::WriteAllText((Join-Path $PSScriptRoot ".dev_session_id"), $sessionId)
}

Write-Host "3. Sending chat prompt: '$Message'..." -ForegroundColor Cyan
$chatBody = @{
    session_id = $sessionId
    message    = $Message
} | ConvertTo-Json

$chatRes = Invoke-RestMethod -Uri "$baseUrl/chat" -Method Post -Headers $apiHeaders -Body $chatBody
$chatRes | ConvertTo-Json -Depth 5