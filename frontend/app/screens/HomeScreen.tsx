import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { 
  View, 
  Text, 
  TouchableOpacity, 
  ScrollView, 
  ActivityIndicator, 
  Linking,
} from "react-native";
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as knn from "@tensorflow-models/knn-classifier";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const isWeb = typeof document !== "undefined";

type Pred = { label: string; confidences: Record<string, number> } | null;

function logTime() {
  const d = new Date();
  return [d.getHours().toString().padStart(2, "0"), d.getMinutes().toString().padStart(2, "0"), d.getSeconds().toString().padStart(2, "0")].join(":");
}

async function fileToImage(file: File): Promise<HTMLImageElement> {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  await new Promise((res, rej) => {
    img.onload = () => res(null);
    img.onerror = rej;
  });
  return img; // caller keeps the objectURL alive for previews
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <Text style={{ fontWeight: "700", fontSize: 18, marginBottom: 8 }}>{children}</Text>;
}

function Pill({ children }: { children: React.ReactNode }) {
  return (
    <View style={{ backgroundColor: "#eef2ff", borderColor: "#c7d2fe", borderWidth: 1, paddingHorizontal: 8, paddingVertical: 4, borderRadius: 999 }}>
      <Text style={{ fontSize: 12, fontWeight: "600" }}>{children}</Text>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
export default function HomeScreen() {
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState<string | null>(null);
  const [messages, setMessages] = useState<string[]>([]);
  const [net, setNet] = useState<mobilenet.MobileNet | null>(null);
  const clfRef = useRef<knn.KNNClassifier | null>(null);
  const [classCounts, setClassCounts] = useState<Record<string, number>>({});

  // Imported-but-not-yet-learned files
  const [pendingFiles, setPendingFiles] = useState<Array<{ file: File; label: string; uri: string }>>([]);

  // Previews (shows Train images regardless of learned or not)
  const [trainPreviews, setTrainPreviews] = useState<Array<{ uri: string; label: string }>>([]);

  // Test image & prediction
  const [testPreview, setTestPreview] = useState<string | null>(null);
  const [testFile, setTestFile] = useState<File | null>(null);
  const [pred, setPred] = useState<Pred>(null);

  const pushMsg = useCallback((s: string) => setMessages((m) => [`[${logTime()}] ${s}`, ...m].slice(0, 200)), []);

  useEffect(() => {
    (async () => {
      try {
        setLoading("Preparing TensorFlow.js backend...");
        if (tf.getBackend() !== "webgl" && (tf as any).engine?.registryFactory?.["webgl"]) {
          await tf.setBackend("webgl");
        }
        await tf.ready();
        pushMsg(`TFJS backend: ${tf.getBackend()}`);
        setLoading("Loading MobileNet (feature extractor)...");
        const model = await mobilenet.load({ version: 2, alpha: 1.0 });
        setNet(model);
        clfRef.current = knn.create();
        setReady(true);
        setLoading(null);
        pushMsg("Ready. Import a folder structured as /<label>/<image>. Then click Learn and Predict.");
      } catch (e: any) {
        setLoading(null);
        pushMsg(`[ERROR] ${e?.message || e}`);
      }
    })();
  }, [pushMsg]);

  // -------------------------------------------------------------------------
  // Folder Import: root/.../<label>/<image>
  // -------------------------------------------------------------------------
  const onAddFolder = useCallback(async (files: FileList | null) => {
    if (!files) return;
    setLoading("Scanning folder...");

    const newPending: Array<{ file: File; label: string; uri: string }> = [];
    const previews: Array<{ uri: string; label: string }> = [];
    let added = 0;

    try {
      for (let i = 0; i < files.length; i++) {
        const f = files[i];
        if (!(f.type && f.type.startsWith("image/"))) continue;
        const rel = (f as any).webkitRelativePath || f.name; // e.g., animals/cat/img1.jpg or img1.jpg
        const parts = rel.split("/").filter(Boolean);
        // Use the parent folder name as class label: .../<label>/<file>
        const label = parts.length >= 2 ? parts[parts.length - 2] : "root";

        const uri = URL.createObjectURL(f);
        newPending.push({ file: f, label, uri });
        previews.push({ uri, label });
        added++;
      }

      if (added === 0) {
        pushMsg("No images found in the selected folder.");
      } else {
        setPendingFiles((prev) => [...newPending, ...prev]);
        setTrainPreviews((prev) => [...previews, ...prev].slice(0, 200));
        pushMsg(`Imported ${added} image(s) pending for learning.`);
      }
    } catch (e: any) {
      pushMsg(`[ERROR] folder import: ${e?.message || e}`);
    } finally {
      setLoading(null);
    }
  }, [pushMsg]);

  // -------------------------------------------------------------------------
  // Learn (process pendingFiles into KNN using MobileNet features)
  // -------------------------------------------------------------------------
  const onLearn = useCallback(async () => {
    if (!net || !clfRef.current) return pushMsg("Model not ready yet");
    if (pendingFiles.length === 0) return pushMsg("No pending images to learn.");

    setLoading(`Learning ${pendingFiles.length} image(s)...`);
    try {
      // Add examples in small batches to keep UI responsive
      const BATCH = 24;
      for (let i = 0; i < pendingFiles.length; i++) {
        const { file, label } = pendingFiles[i];
        const img = await fileToImage(file);
        const logits = tf.tidy(() => net.infer(img, true) as tf.Tensor);
        clfRef.current.addExample(logits, label);
        logits.dispose();
        if (i % BATCH === 0) await tf.nextFrame();
      }
      const counts = clfRef.current.getClassExampleCount();
      setClassCounts(counts as any);
      pushMsg(`Learned. Classes: ${Object.keys(counts).join(", ") || "(none)"}`);
      setPendingFiles([]);
    } catch (e: any) {
      pushMsg(`[ERROR] learn: ${e?.message || e}`);
    } finally {
      setLoading(null);
    }
  }, [net, pendingFiles, pushMsg]);

  // -------------------------------------------------------------------------
  // Test image selection & Predict button
  // -------------------------------------------------------------------------
  const onSelectTest = useCallback(async (file: File | null) => {
    setPred(null);
    setTestFile(file);
    if (file) setTestPreview(URL.createObjectURL(file));
    else setTestPreview(null);
  }, []);

  const onPredictBtn = useCallback(async () => {
    if (!testFile) return pushMsg("Pick a test image first.");
    if (!net || !clfRef.current) return pushMsg("Model not ready yet");
    if (Object.keys(clfRef.current.getClassExampleCount()).length === 0) {
      return pushMsg("No classes yet. Click Learn after importing a folder.");
    }
    setLoading("Running inference...");
    setPred(null);
    try {
      const img = await fileToImage(testFile);
      const logits = tf.tidy(() => net.infer(img, true) as tf.Tensor);
      const res = await clfRef.current.predictClass(logits, 5);
      logits.dispose();
      setPred({ label: res.label, confidences: (res.confidences as any) || {} });
      pushMsg(`Prediction: ${res.label}`);
    } catch (e: any) {
      pushMsg(`[ERROR] predict: ${e?.message || e}`);
    } finally {
      setLoading(null);
    }
  }, [net, testFile, pushMsg]);

  const onClear = useCallback(() => {
    clfRef.current = knn.create();
    setClassCounts({});
    setPendingFiles([]);
    setTrainPreviews([]);
    setTestPreview(null);
    setTestFile(null);
    setPred(null);
    pushMsg("Cleared classifier, pending files, and previews.");
  }, [pushMsg]);

  const REPO_URL = "https://github.com/europanite/client_side_ai_training";

  // -------------------------------------------------------------------------
  // UI helpers
  // -------------------------------------------------------------------------
  const FilePickFolder = () => {
    if (!isWeb) return null as any;

    const ref = useRef<HTMLInputElement | null>(null);

    // Chromium/Safari/Firefox
    useEffect(() => {
      if (!ref.current) return;
      ref.current.setAttribute("webkitdirectory", "");
      ref.current.setAttribute("directory", "");
      ref.current.setAttribute("mozdirectory", "");
    }, []);

    return (
      <input
        ref={ref}
        type="file"
        multiple
        style={{ marginRight: 8, marginTop: 4, marginBottom: 4 }}
        onChange={(e: any) => onAddFolder(e.target.files as FileList)}
      />
    );
  };

  const FilePickTest = () => {
    if (!isWeb) return null as any;
    return (
      // @ts-ignore
      <input
        type="file"
        accept="image/*"
        style={{ marginRight: 8, marginTop: 4, marginBottom: 4 }}
        onChange={(e: any) => onSelectTest((e.target.files as FileList)?.[0] || null)}
      />
    );
  };

  const ClassCountBadges = useMemo(() => {
    const entries = Object.entries(classCounts);
    if (entries.length === 0) return <Text style={{ color: "#666" }}>(No learned examples yet)</Text>;
    return (
      <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 6 }}>
        {entries.map(([label, n]) => (
          <Pill key={label}>{label}: {n}</Pill>
        ))}
      </View>
    );
  }, [classCounts]);

  const groupedPreviews = useMemo(() => {
    // Map<label, string[] (uris)>
    const m = new Map<string, string[]>();
    for (const p of trainPreviews) {
      const arr = m.get(p.label) ?? [];
      arr.push(p.uri);
      m.set(p.label, arr);
    }
    return Array.from(m.entries());
  }, [trainPreviews]);
  return (
    <ScrollView style={{ flex: 1, backgroundColor: "#f8fafc" }} contentContainerStyle={{ padding: 16 }}>
      <TouchableOpacity onPress={() => Linking.openURL(REPO_URL)}>
        <Text
          style={{
            fontSize: 24,
            fontWeight: "800",
            marginBottom: 12,
            color: "#1d4ed8",
            textDecorationLine: "underline",
          }}
        >
          Client Side AI Training
        </Text>
      </TouchableOpacity>
      <View style={{ marginBottom: 16, padding: 12, borderRadius: 12, borderWidth: 1, backgroundColor: "#fff" }}>
        <SectionTitle>Status</SectionTitle>
        <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
          {ready ? <Pill>Ready</Pill> : <Pill>Loading</Pill>}
          {loading && (
            <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
              <ActivityIndicator />
              <Text>{loading}</Text>
            </View>
          )}
        </View>
      </View>

      {/* Step 1: Import train data */}
      <View style={{ marginBottom: 16, padding: 12, borderRadius: 12, borderWidth: 1, backgroundColor: "#fff" }}>
        <SectionTitle>1. Train </SectionTitle>
        <Text style={{ marginBottom: 8 }}>
          Please select the top folder. The directory name above each image will be used as the class name.
        </Text>
        <View style={{ flexDirection: "row", alignItems: "center", gap: 8, marginBottom: 8 }}>
          <FilePickFolder />
          <TouchableOpacity onPress={onLearn} style={{ backgroundColor: "#dcfce7", borderWidth: 1, borderColor: "#bbf7d0", paddingHorizontal: 12, paddingVertical: 8, borderRadius: 8 }}>
            <Text style={{ fontWeight: "800", color: "#166534" }}>Learn</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={onClear} style={{ backgroundColor: "#fee2e2", borderWidth: 1, borderColor: "#fecaca", paddingHorizontal: 10, paddingVertical: 8, borderRadius: 8 }}>
            <Text style={{ fontWeight: "700", color: "#991b1b" }}>Clear</Text>
          </TouchableOpacity>
        </View>
        <Text style={{ color: "#475569", marginBottom: 6 }}>Pending images: {pendingFiles.length}</Text>
        <View style={{ marginTop: 6 }}>{ClassCountBadges}</View>
      </View>

      {/* Step 2: Pick test & Predict */}
      <View style={{ marginBottom: 16, padding: 12, borderRadius: 12, borderWidth: 1, backgroundColor: "#fff" }}>
        <SectionTitle>2) Test </SectionTitle>
        <Text style={{ marginBottom: 8 }}>Select a test image and then press "Predict".</Text>
        <View style={{ flexDirection: "row", alignItems: "center", gap: 8, marginBottom: 8 }}>
          <FilePickTest />
          <TouchableOpacity onPress={onPredictBtn} style={{ backgroundColor: "#e0f2fe", borderWidth: 1, borderColor: "#bae6fd", paddingHorizontal: 12, paddingVertical: 8, borderRadius: 8 }}>
            <Text style={{ fontWeight: "800", color: "#075985" }}>Predict</Text>
          </TouchableOpacity>
        </View>

        {/* Previews */}
        <View style={{ gap: 8, marginTop: 8 }}>
          {testPreview && (
            <View>
              <Text style={{ fontWeight: "700", marginBottom: 6 }}>Test Image</Text>
              {/* @ts-ignore web-only */}
              <img src={testPreview} alt="test" style={{ maxWidth: 320, borderRadius: 8, border: "1px solid #ddd" }} />
            </View>
          )}
          {pred && (
            <View>
              <Text style={{ fontWeight: "700", marginBottom: 6 }}>Prediction</Text>
              <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 6 }}>
                <Pill>Top: {pred.label}</Pill>
                {Object.entries(pred.confidences)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 5)
                  .map(([k, v]) => (
                    <Pill key={k}>{k}: {(v * 100).toFixed(1)}%</Pill>
                  ))}
              </View>
            </View>
          )}
        </View>
      </View>

      {/* Previews of Train images */}
      <View style={{ marginBottom: 16, padding: 12, borderRadius: 12, borderWidth: 1, backgroundColor: "#fff" }}>
        <SectionTitle>Training Data Previews</SectionTitle>

        {groupedPreviews.length === 0 ? (

          <Text style={{ color: "#666" }}>(Nothing yet)</Text>
        ) : (
          <View style={{ gap: 12 }}>
            {groupedPreviews.map(([label, uris]) => (
              <View key={label} style={{}}>
                <Text style={{ fontWeight: "700", marginBottom: 6 }}>{label}</Text>
                <ScrollView
                  horizontal
                  showsHorizontalScrollIndicator={false}
                  style={{}}
                  contentContainerStyle={{ flexDirection: "row", alignItems: "center", gap: 8 }}
                >
                  {uris.map((uri, i) => (
                    <View key={i} style={{ alignItems: "center" }}>
                      {/* @ts-ignore web-only */}
                      <img
                        src={uri}
                        alt={label}
                        style={{ width: 96, height: 96, objectFit: "cover", borderRadius: 8, border: "1px solid #e5e7eb" }}
                      />
                    </View>
                  ))}
                </ScrollView>
              </View>
            ))}
          </View>
        )}
      </View>

      <View style={{ marginBottom: 32, padding: 12, borderRadius: 12, borderWidth: 1, backgroundColor: "#fff" }}>
        <SectionTitle>Console</SectionTitle>
        <View style={{ backgroundColor: "#0b1220", borderRadius: 8, padding: 10 }}>
          {messages.length === 0 ? (
            <Text style={{ color: "#9ca3af", fontFamily: "monospace" }}>(No output yet.)</Text>
          ) : (
            messages.map((m, i) => (
              <Text key={i} style={{ color: "#e5e7eb", fontFamily: "monospace", marginBottom: 4 }}>{m}</Text>
            ))
          )}
        </View>
      </View>

      <Text style={{ color: "#64748b", fontSize: 12, marginBottom: 24 }}>
        Notes: Folder format is <code>.../&lt;label&gt;/&lt;image files&gt;</code>. Select a folder, click <b>Learn</b>, then choose a test image and click <b>Predict</b>. Images never leave your device.
      </Text>
    </ScrollView>
  );
}

