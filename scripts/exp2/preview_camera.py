"""
RealSense ライブプレビュー
RGB + depth をリアルタイム表示。位置・距離調整用。

  python scripts/exp2/preview_camera.py

操作:
  q / Esc : 終了
  s       : 現在フレームを /tmp/preview_snap.png に保存
"""
import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

print("プレビュー起動  q/Esc=終了  s=スナップショット保存")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color = np.asanyarray(aligned.get_color_frame().get_data())
        depth_raw = np.asanyarray(aligned.get_depth_frame().get_data())
        depth_m = depth_raw * depth_scale

        # depth カラーマップ (0.2〜0.7m)
        depth_vis = np.clip((depth_m - 0.2) / 0.5, 0, 1)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # 中央の depth 値をオーバーレイ
        h, w = depth_m.shape
        cx, cy = w // 2, h // 2
        center_d = depth_m[cy-10:cy+10, cx-10:cx+10]
        center_d = center_d[center_d > 0.05]
        d_text = f"{np.median(center_d)*1000:.0f}mm" if len(center_d) else "---"
        cv2.putText(color, f"center: {d_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.circle(color, (cx, cy), 10, (0, 255, 0), 2)

        combined = np.hstack([color, depth_color])
        cv2.imshow("RealSense Preview  (q=quit  s=snap)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            cv2.imwrite("/tmp/preview_snap.png", color)
            print("スナップ保存: /tmp/preview_snap.png")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
