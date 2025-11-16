@echo off
REM Windows batch script for building .exe file
REM Windows için .exe oluşturma scripti

echo ========================================
echo Image Processing App - EXE Builder
echo Görüntü İşleme Uygulaması - EXE Oluşturucu
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo HATA: Python bulunamadi!
    echo Lütfen Python'u yükleyin.
    pause
    exit /b 1
)

echo Python bulundu.
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo PyInstaller bulunamadi. Yükleniyor...
    pip install pyinstaller
    if errorlevel 1 (
        echo HATA: PyInstaller yüklenemedi!
        pause
        exit /b 1
    )
)

echo PyInstaller hazir.
echo.

REM Install requirements
echo Bağımlılıklar kontrol ediliyor...
pip install -r requirements.txt
if errorlevel 1 (
    echo UYARI: Bazı bağımlılıklar yüklenemedi!
)

echo.
echo ========================================
echo EXE dosyasi olusturuluyor...
echo ========================================
echo.

REM Build executable
pyinstaller build_exe.spec --clean

if errorlevel 1 (
    echo.
    echo HATA: EXE dosyasi olusturulamadi!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Basarili!
echo ========================================
echo.
echo EXE dosyasi: dist\ImageProcessingApplication.exe
echo.
echo Test etmek icin dist klasorune gidin ve
echo ImageProcessingApplication.exe dosyasini calistirin.
echo.
pause
