import { useCallback, useEffect, useRef, useState } from "react";
import {
  EuiBadge, EuiButton, EuiButtonEmpty, EuiCallOut, EuiFlexGrid, EuiFlexGroup, EuiFlexItem,
  EuiHealth, EuiImage, EuiPageTemplate, EuiPanel, EuiProgress, EuiSpacer, EuiText, EuiTitle,
  useEuiTheme,
} from "@elastic/eui";
import { startDetector } from "./detector.js";
import { LETTERS, STABILIZER_DEFAULTS } from "./sibi.js";

const IDLE_READING = { letter: null, conf: 0, progress: 0, hasHand: false };

function errorMessage(err) {
  if (err?.name === "NotAllowedError") {
    return "Izin kamera ditolak. Izinkan kamera untuk situs ini di pengaturan browser, lalu muat ulang halaman.";
  }
  if (err?.name === "NotFoundError") return "Kamera tidak ditemukan. Sambungkan webcam lalu coba lagi.";
  return `Gagal memulai: ${err?.message ?? err}. Coba muat ulang halaman, atau pakai Chrome/Edge terbaru.`;
}

function CameraPanel({ onReading, onEmit }) {
  const { euiTheme } = useEuiTheme();
  const canvasRef = useRef(null);
  const stopRef = useRef(null);
  const [stage, setStage] = useState(null); // null = idle, string = loading, "running"
  const [error, setError] = useState(null);
  const [hasHand, setHasHand] = useState(false);

  useEffect(() => () => stopRef.current?.(), []);

  const start = useCallback(async () => {
    setError(null);
    try {
      stopRef.current = await startDetector(canvasRef.current, (r) => {
        onReading(r);
        setHasHand(r.hasHand);
        if (r.emitted) onEmit(r.emitted);
      }, setStage);
      setStage("running");
    } catch (err) {
      console.error(err);
      setError(errorMessage(err));
      setStage(null);
    }
  }, [onReading, onEmit]);

  const running = stage === "running";
  return (
    <EuiPanel hasBorder paddingSize="m">
      <div style={{ position: "relative", aspectRatio: "4 / 3", maxWidth: "100%", borderRadius: euiTheme.border.radius.medium,
        overflow: "hidden", background: euiTheme.colors.backgroundBaseSubdued }}>
        <canvas ref={canvasRef} width={640} height={480} style={{ width: "100%", height: "100%", display: "block" }} />
        {!running && (
          <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", padding: euiTheme.size.l, textAlign: "center" }}>
            <div style={{ maxWidth: 360 }}>
              <EuiTitle size="xs"><h2>Kamera belum aktif</h2></EuiTitle>
              <EuiSpacer size="s" />
              <EuiText size="s" color="subdued">
                <p>{stage ?? "Video hanya diproses di browsermu. Tidak ada gambar yang dikirim ke server."}</p>
              </EuiText>
              <EuiSpacer size="m" />
              <EuiButton fill iconType="videoPlayer" onClick={start} isLoading={stage !== null}>
                Nyalakan kamera
              </EuiButton>
            </div>
          </div>
        )}
      </div>
      {error && (<><EuiSpacer size="m" /><EuiCallOut color="danger" iconType="error" size="s" title={error} /></>)}
      <EuiSpacer size="s" />
      <EuiHealth color={running && hasHand ? "success" : "subdued"}>
        {running ? (hasHand ? "Tangan terdeteksi" : "Tidak ada tangan") : "Kamera mati"}
      </EuiHealth>
    </EuiPanel>
  );
}

function ReadingPanel({ reading, text, setText }) {
  const { euiTheme } = useEuiTheme();
  const confPct = Math.round(reading.conf * 100);
  const lowConf = reading.letter && reading.conf < STABILIZER_DEFAULTS.minConf;
  return (
    <EuiPanel hasBorder paddingSize="l">
      <EuiFlexGroup alignItems="center" gutterSize="l" responsive={false}>
        <EuiFlexItem grow={false}>
          <div aria-live="polite" style={{ width: "1.2em", textAlign: "center", fontSize: "5rem", lineHeight: 1,
            fontWeight: euiTheme.font.weight.bold, color: reading.letter ? euiTheme.colors.primary : euiTheme.colors.textDisabled }}>
            {reading.letter ?? "–"}
          </div>
        </EuiFlexItem>
        <EuiFlexItem>
          <EuiProgress label="Yakin" valueText={`${confPct}%`} value={confPct} max={100} size="m"
            color={lowConf ? "warning" : "primary"} />
          <EuiSpacer size="m" />
          <EuiProgress label="Tahan" valueText={`${Math.round(reading.progress * 100)}%`}
            value={Math.round(reading.progress * 100)} max={100} size="m" color="success" />
        </EuiFlexItem>
      </EuiFlexGroup>

      <EuiSpacer size="l" />
      <EuiTitle size="xxs"><h3>Teks</h3></EuiTitle>
      <EuiSpacer size="xs" />
      <EuiPanel color="subdued" hasShadow={false} paddingSize="m">
        <EuiText style={{ fontFamily: euiTheme.font.familyCode, fontSize: "1.5rem", minHeight: "2.25rem", wordBreak: "break-all" }}>
          <span>{text || <span style={{ color: euiTheme.colors.textDisabled, fontFamily: euiTheme.font.family, fontSize: euiTheme.size.m }}>
            Huruf yang terbaca muncul di sini</span>}</span>
        </EuiText>
      </EuiPanel>
      <EuiSpacer size="s" />
      <EuiFlexGroup gutterSize="s" wrap responsive={false}>
        <EuiFlexItem grow={false}><EuiButton size="s" onClick={() => setText((t) => t + " ")}>Spasi</EuiButton></EuiFlexItem>
        <EuiFlexItem grow={false}><EuiButton size="s" iconType="arrowLeft" onClick={() => setText((t) => t.slice(0, -1))}>Hapus</EuiButton></EuiFlexItem>
        <EuiFlexItem grow={false}><EuiButtonEmpty size="s" color="danger" iconType="trash" onClick={() => setText("")}>Bersihkan</EuiButtonEmpty></EuiFlexItem>
      </EuiFlexGroup>

      <EuiSpacer size="l" />
      <EuiTitle size="xxs"><h3>Huruf yang dikenali</h3></EuiTitle>
      <EuiSpacer size="s" />
      <EuiFlexGrid columns={4} gutterSize="s" responsive={false} style={{ gridTemplateColumns: "repeat(auto-fill, minmax(2.25rem, 1fr))" }}>
        {LETTERS.map((l) => (
          <EuiBadge key={l} color={l === reading.letter ? "primary" : "hollow"} style={{ justifyContent: "center" }}>{l}</EuiBadge>
        ))}
      </EuiFlexGrid>
    </EuiPanel>
  );
}

function AlphabetGuide() {
  return (
    <EuiPanel hasBorder paddingSize="l">
      <EuiTitle size="xxs"><h3>Panduan abjad SIBI</h3></EuiTitle>
      <EuiSpacer size="s" />
      <EuiImage src="/abjad-sibi.png" alt="Bentuk tangan abjad SIBI dari A sampai Z"
        allowFullScreen size="fullWidth" style={{ maxWidth: 412, margin: "0 auto", display: "block" }} />
    </EuiPanel>
  );
}

export default function App() {
  const [reading, setReading] = useState(IDLE_READING);
  const [text, setText] = useState("");
  const onEmit = useCallback((letter) => setText((t) => t + letter), []);

  return (
    <EuiPageTemplate restrictWidth={1200} panelled={false} grow>
      <EuiPageTemplate.Header pageTitle="Penerjemah Abjad SIBI" />
      <EuiPageTemplate.Section>
        <EuiFlexGroup gutterSize="l" alignItems="flexStart">
          <EuiFlexItem grow={3}><CameraPanel onReading={setReading} onEmit={onEmit} /></EuiFlexItem>
          <EuiFlexItem grow={2}>
            <ReadingPanel reading={reading} text={text} setText={setText} />
            <EuiSpacer size="l" />
            <AlphabetGuide />
          </EuiFlexItem>
        </EuiFlexGroup>
      </EuiPageTemplate.Section>
    </EuiPageTemplate>
  );
}
