import { useEffect, useState } from "react";
import { Alert, Button, Drawer, FileButton, Group, Loader } from "@mantine/core";
import { Dropzone } from "@mantine/dropzone";
import { useDisclosure } from "@mantine/hooks";
import { SettingsSidebar } from "./components/SettingsSidebar";
import { StepsPanel } from "./components/StepsPanel";
import { Workspace } from "./components/Workspace";
import { useHealth, useOptions } from "./hooks/useBackend";
import { useDetect } from "./hooks/useDetect";
import { usePrepare } from "./hooks/usePrepare";
import { useSettings } from "./store/settings";

const ACCEPT = ["image/jpeg", "image/png", "image/webp"];

function useObjectUrl(file: File | null) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!file) return setUrl(null);
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  return url;
}

export default function App() {
  useOptions(); // chọn engine mặc định ngay cả khi ngăn cài đặt đang đóng
  const { isError, warming } = useHealth();
  const { method, device, findPoles } = useSettings();
  const [file, setFile] = useState<File | null>(null);
  const [settingsOpen, settings] = useDisclosure(false);
  const preview = useObjectUrl(file);

  const prep = usePrepare(method || undefined, device);
  const poles = usePrepare("poles", device, findPoles);
  const ready = prep.ready && poles.ready;
  const prepFailed = prep.state === "error" || poles.state === "error";

  const { sid, det, job, state, error } = useDetect(file, poles.ready);
  const detecting = !!file && state !== "error" && !det;

  const preparing = warming || !ready;
  const status = isError
    ? "Mất kết nối máy chủ"
    : prepFailed
      ? "Nạp model lỗi, mở Cài đặt để thử lại"
      : preparing
        ? "Đang chuẩn bị model"
        : "Sẵn sàng";

  return (
    <div className="app">
      <header className="bar">
        <span className="wordmark">RunnerPop</span>
        <Group gap="md" wrap="nowrap">
          <span className="status" role="status">
            {preparing && !isError && !prepFailed ? <Loader size={12} color="ink" /> : <i className="status-dot" data-error={isError || prepFailed} />}
            <span className="status-text">{status}</span>
          </span>
          {file && (
            <FileButton onChange={setFile} accept={ACCEPT.join(",")}>
              {(props) => <Button {...props} size="xs" variant="default">Đổi ảnh</Button>}
            </FileButton>
          )}
          <Button size="xs" variant="default" onClick={settings.open}>Cài đặt</Button>
        </Group>
      </header>

      {!file ? (
        <div className="empty">
          <Dropzone className="drop" onDrop={(f) => setFile(f[0] ?? null)} accept={ACCEPT} multiple={false} disabled={!ready}>
            <p className="drop-title">{ready ? "Thả ảnh người chạy vào đây" : "Đang chuẩn bị model"}</p>
            <p className="drop-sub">
              {ready ? "Hoặc bấm để chọn ảnh JPG, PNG, WebP. Người chạy giữ màu, nền về đen trắng." : "Lần đầu có thể mất vài phút. Tải ảnh được ngay khi xong."}
            </p>
          </Dropzone>
        </div>
      ) : det && sid ? (
        <Workspace key={sid} sid={sid} det={det} detectStages={job.stages} ready={ready} />
      ) : (
        <div className="layout">
          <div className="main">
            {state === "error" && <Alert color="red">{error} Bấm Đổi ảnh để thử ảnh khác.</Alert>}
            {preview && (
              <div className="stage">
                <div className="frame" data-dim="true">
                  <img src={preview} alt="Ảnh vừa tải lên" />
                  {detecting && (
                    <div className="busy">
                      <Loader size={14} color="ink" />
                      {poles.ready ? job.running ?? "Đang tải ảnh lên" : "Đang nạp model tìm gậy"}
                    </div>
                  )}
                </div>
              </div>
            )}
            <StepsPanel detect={job.stages} detectRunning={job.running} />
          </div>
          <aside className="rail">
            <div className="rail-body">
              <div>
                <h3 className="rail-title">Người chạy</h3>
                <p className="rail-note">{detecting ? "Đang tìm người trong ảnh." : "Chưa có kết quả."}</p>
              </div>
            </div>
          </aside>
        </div>
      )}

      <Drawer opened={settingsOpen} onClose={settings.close} position="right" size={380} title="Cài đặt">
        <SettingsSidebar />
      </Drawer>
    </div>
  );
}
