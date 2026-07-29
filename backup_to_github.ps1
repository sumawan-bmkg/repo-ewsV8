<# ===================================================================
   backup_to_github.ps1
   Skrip otomatisasi Git — Backup Disertasi EWS ke GitHub
   ===================================================================
   Cara pakai:
     1. Buka PowerShell sebagai Administrator (jika perlu).
     2. cd D:\multi\scalogramv3\disertasi4
     3. .\backup_to_github.ps1
     4. Masukkan URL repositori GitHub saat diminta.
   =================================================================== #>

Write-Host ""
Write-Host "============================================================"
Write-Host "  BACKUP DISERTASI EWS GEMPA BUMI V8 SUPCON KE GITHUB"
Write-Host "============================================================"
Write-Host ""

# ─── LANGKAH 1: Cek status Git ─────────────────────────────────
Write-Host "[1] Memeriksa status Git..." -ForegroundColor Cyan

if (Test-Path ".git") {
    Write-Host "    [OK] Repositori Git sudah ada." -ForegroundColor Green
} else {
    Write-Host "    [..] Belum ada repositori. Menjalankan git init..." -ForegroundColor Yellow
    git init
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    [ERROR] Gagal inisialisasi Git. Pastikan Git terinstal." -ForegroundColor Red
        exit 1
    }
    Write-Host "    [OK] git init berhasil." -ForegroundColor Green
}

# ─── LANGKAH 2: Cek .gitignore ─────────────────────────────────
Write-Host ""
Write-Host "[2] Memeriksa file .gitignore..." -ForegroundColor Cyan

if (Test-Path ".gitignore") {
    Write-Host "    [OK] .gitignore ditemukan." -ForegroundColor Green
} else {
    Write-Host "    [WARNING] .gitignore tidak ditemukan."
    Write-Host "    Sebaiknya buat .gitignore sebelum melanjutkan."
    Write-Host "    (lihat template di akhir dokumentasi)"
}

# ─── LANGKAH 3: Cek ukuran file .pth ───────────────────────────
Write-Host ""
Write-Host "[3] Memeriksa ukuran file bobot model (*.pth)..." -ForegroundColor Cyan

$largeFiles = Get-ChildItem -Recurse -Filter "*.pth" | Where-Object { $_.Length -gt 100MB }
if ($largeFiles.Count -gt 0) {
    Write-Host "    [WARNING] Ditemukan file .pth > 100 MB:" -ForegroundColor Yellow
    foreach ($f in $largeFiles) {
        $sizeMB = [math]::Round($f.Length / 1MB, 1)
        Write-Host "      - $($f.FullName) ($sizeMB MB)" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "    File > 100 MB akan ditolak GitHub (kecuali pakai Git LFS)."
    Write-Host "    Untuk mengaktifkan Git LFS, jalankan:"
    Write-Host '      git lfs track "*.pth"'
    Write-Host "    lalu commit file .gitattributes yang dihasilkan."
    Write-Host ""
    $continue = Read-Host "    Tetap lanjutkan? (y/n) [default: n]"
    if ($continue -ne "y") {
        Write-Host "    [STOP] Hentikan proses. Aktifkan LFS dulu." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "    [OK] Semua file .pth di bawah 100 MB (aman tanpa LFS)." -ForegroundColor Green
}

# ─── LANGKAH 4: Git add ────────────────────────────────────────
Write-Host ""
Write-Host "[4] Menambahkan file ke staging (git add --all)..." -ForegroundColor Cyan

git add --all
if ($LASTEXITCODE -ne 0) {
    Write-Host "    [ERROR] Gagal git add." -ForegroundColor Red
    exit 1
}

# Tampilkan ringkasan
$fileCount = (git diff --cached --name-only).Count
Write-Host "    $fileCount file akan di-commit." -ForegroundColor Green

# ─── LANGKAH 5: Git commit ─────────────────────────────────────
Write-Host ""
Write-Host "[5] Membuat commit..." -ForegroundColor Cyan

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
$commitMessage = "feat(disertasi-ews): archive Blind Test 2026 forensic evidence, 3-Phase optimization scripts, and final calibration diagrams

- Temperature Scaling (T=6.3) post-processing calibration
- Class-Weighted Focal Loss training resume (Phase 2)
- Decoupled Training architecture optimization (Phase 3)
- Blind Test 2026 comparison: Recall 12.9% -> 91.5%
- Deployment package for BMKG production server

Auto-generated: $timestamp"

git commit -m $commitMessage
if ($LASTEXITCODE -ne 0) {
    Write-Host "    [ERROR] Gagal commit. Mungkin tidak ada perubahan." -ForegroundColor Red
    Write-Host "    Jika tidak ada perubahan baru, jalankan ulang setelah ada modifikasi."
    exit 1
}
Write-Host "    [OK] Commit berhasil." -ForegroundColor Green

# ─── LANGKAH 6: Setup remote ───────────────────────────────────
Write-Host ""
Write-Host "[6] Konfigurasi remote origin..." -ForegroundColor Cyan

# Cek apakah remote sudah ada
$existingRemote = git remote get-url origin 2>$null
if ($existingRemote) {
    Write-Host "    Remote origin sudah terdaftar: $existingRemote"
    $changeRemote = Read-Host "    Ganti remote? (y/n) [default: n]"
    if ($changeRemote -eq "y") {
        git remote remove origin
        $repoUrl = Read-Host "    Masukkan URL repositori GitHub (contoh: https://github.com/username/repo-ews.git)"
        if ([string]::IsNullOrWhiteSpace($repoUrl)) {
            Write-Host "    [ERROR] URL tidak boleh kosong." -ForegroundColor Red
            exit 1
        }
        git remote add origin $repoUrl
        Write-Host "    [OK] Remote origin diperbarui." -ForegroundColor Green
    }
} else {
    Write-Host "    Belum ada remote origin."
    $repoUrl = Read-Host "    Masukkan URL repositori GitHub (contoh: https://github.com/username/repo-ews.git)"
    if ([string]::IsNullOrWhiteSpace($repoUrl)) {
        Write-Host "    [ERROR] URL tidak boleh kosong." -ForegroundColor Red
        exit 1
    }
    git remote add origin $repoUrl
    Write-Host "    [OK] Remote origin ditambahkan." -ForegroundColor Green
}

# ─── LANGKAH 7: Push ke GitHub ─────────────────────────────────
Write-Host ""
Write-Host "[7] Push ke GitHub (git push -u origin main)..." -ForegroundColor Cyan

git branch -M main
git push -u origin main
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "    [ERROR] Push gagal. Kemungkinan penyebab:" -ForegroundColor Red
    Write-Host "      1. URL remote salah. Cek dengan: git remote -v"
    Write-Host "      2. Belum login GitHub. Jalankan: gh auth login"
    Write-Host "      3. Repositori tujuan belum dibuat di GitHub."
    Write-Host "      4. Ada file terlalu besar (>100 MB). Cek LFS."
    Write-Host "      5. Koneksi internet terputus."
    Write-Host ""
    Write-Host "    Setelah masalah diperbaiki, jalankan ulang skrip."
    exit 1
}

# ─── SELESAI ────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================"
Write-Host "  BACKUP SELESAI!" -ForegroundColor Green
Write-Host "============================================================"
Write-Host ""
Write-Host "  Repositori berhasil di-push ke GitHub."
Write-Host "  Seluruh artefak riset aman di cloud."
Write-Host ""
Write-Host "  Link repositori:"
Write-Host "    https://github.com/<username>/<repo-name>"
Write-Host ""
Write-Host "  Langkah selanjutnya (setelah push):"
Write-Host "    1. Buka repositori di browser GitHub."
Write-Host "    2. Edit file README.md (template sudah disediakan)."
Write-Host "    3. Atur visibilitas: Settings -> General -> Visibility."
Write-Host "    4. (Opsional) Aktifkan GitHub Pages untuk README yang lebih rapi."
Write-Host ""
Write-Host "============================================================"
Write-Host ""
