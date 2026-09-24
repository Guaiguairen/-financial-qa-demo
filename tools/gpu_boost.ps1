# 解除联想固件对 GPU 功率的异常限制（cTGP 20W → 60W）
# 用法：右键"使用 PowerShell 运行"，或： powershell -ExecutionPolicy Bypass -File tools\gpu_boost.ps1
# 说明：2026-09-24 发现本机 GPU 被固件限制在 20W（机型正常 60W+），
#       通过联想 WMI 接口重写 cTGP/PPAB 功率上限，并清除可能的时钟锁。

Write-Host "[1/3] 写入 cTGP=60 / PPAB=35 ..."
$g = Get-WmiObject -Namespace root\WMI -Class LENOVO_GPU_METHOD
try { $g.GPU_Set_cTGP_PowerLimit(60) | Out-Null; Write-Host "  cTGP 已写入" } catch { Write-Host "  cTGP 写入失败: $($_.Exception.Message)" }
try { $g.GPU_Set_PPAB_PowerLimit(35) | Out-Null; Write-Host "  PPAB 已写入" } catch { Write-Host "  PPAB 写入失败: $($_.Exception.Message)" }

Write-Host "[2/3] 清除 GPU 时钟锁 ..."
nvidia-smi -rgc 2>&1 | Out-Null

Write-Host "[3/3] 当前状态："
$r = $g.GPU_Get_cTGP_PowerLimit()
Write-Host "  EC 侧 cTGP: $($r.Properties['Current_cTGP_PowerLimit'].Value) W (Max=$($r.Properties['Max_cTGP_PowerLimit'].Value))"
nvidia-smi -q -d POWER 2>&1 | Select-String -Pattern 'Current Power Limit' | Select-Object -First 1
nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader
