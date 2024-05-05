# 指定要检查的目录路径
$directory = "out"

# 检查目录是否存在
if (-not (Test-Path -Path $directory -PathType Container)) {
    # 如果目录不存在，则创建它
    New-Item -Path $directory -ItemType Directory -Force
}

nvcc saxpy.cu -o out/saxpy
Write-output "CUDA 编译完成，运行...."
out/saxpy
