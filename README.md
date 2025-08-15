# sharpframes

高质量抽帧工具（阈值 + 比率控制）输出 Linear DNG，适用于 RealityCapture/RealityScan 等 3D 重建数据准备。提供命令行与原生 Qt 界面（PySide6）。

## 安装
```powershell
"D:\ComfyUI123\python\python.exe" -m pip install -r D:\sharpframes\requirements.txt
"D:\ComfyUI123\python\python.exe" -m pip install PySide6
```

## 命令行使用（CLI）
```powershell
"D:\ComfyUI123\python\python.exe" "D:\sharpframes\sharpframes_cli.py" ^
  --input "D:\clothing.MOV" ^
  --output "D:\sharpframes\clothing_out" ^
  --score_threshold 0 ^
  --keep_ratio 0.2 ^
  --min_interval 2 ^
  --save_format dng
```

## 图形界面（Qt）
```powershell
"D:\ComfyUI123\python\python.exe" "D:\sharpframes\sharpframes_gui_qt.py"
```

- 导出为 Linear DNG，如系统看图工具打不开，用 Lightroom/ACR/RawTherapee/darktable 查看。
- 首次运行若需 ffmpeg，imageio-ffmpeg 会自动下载。

参考：
- Sharp Frames Tool: https://sharp-frames.reflct.app/
- nerf_dataset_preprocessing_helper: https://github.com/SharkWipf/nerf_dataset_preprocessing_helper
- FFF_deblur: https://github.com/WalterGropius/FFF_deblur

