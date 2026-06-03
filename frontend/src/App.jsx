import { useState, useEffect } from "react";

function App() {
  const [prediction, setPrediction] = useState(null);
  const [log, setLog] = useState([]);

  useEffect(() => {
    const ws = new WebSocket("wss://audio-intrusion-detection-production.up.railway.app/ws");
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setPrediction(data);
      setLog((prev) => [data, ...prev].slice(0, 20));
    };
    return () => ws.close();
  }, []);

  if (!prediction) return <div className="p-6 text-white bg-gray-900 min-h-screen">Waiting for stream...</div>;

  const breach = prediction.is_breach;

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className={`p-6 rounded-lg mb-6 ${breach ? "bg-red-700" : "bg-green-700"}`}>
        <h1 className="text-2xl font-bold mb-2">
          {breach ? "⚠ BREACH DETECTED" : "PERIMETER SECURE"}
        </h1>
        <p>Class: {prediction.predicted_class}</p>
        <p>Confidence: {(prediction.confidence).toFixed(1)}%</p>
      </div>

      <div>
        <h2 className="text-lg font-semibold mb-3">Event Log</h2>
        {log.map((entry, i) => (
          <div
            key={i}
            className={`p-3 rounded mb-2 text-sm ${entry.is_breach ? "bg-red-900" : "bg-gray-700"}`}
          >
            {entry.predicted_class} — {(entry.confidence).toFixed(1)}% {entry.is_breach ? "⚠" : "✓"}
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;