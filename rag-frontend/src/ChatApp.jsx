import { useState, useEffect, useRef } from "react";
import { Plus, Send, Loader2, Settings, X, Paperclip, MessageSquare, Trash2, FileText } from "lucide-react";
import "./App.css";

const API = "http://localhost:8000";

export default function ChatApp() {
  const [conversations, setConversations] = useState([]);
  const [convId, setConvId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [config, setConfig] = useState(null);
  const [settings, setSettings] = useState({
    embedder: "hf", reranker: "cohere", max_loops: 5, top_k_retrieve: 20, top_n_rerank: 6,
    chunk_size: 1000, chunk_overlap: 200,
  });
  const [docs, setDocs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [ingestJob, setIngestJob] = useState(null);
  const fileRef = useRef(null);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/config`).then(r => r.json()).then(c => {
      setConfig(c);
      setSettings(s => ({ ...s, ...c.defaults }));
    }).catch(() => {});
    refreshConversations();
    refreshDocs();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    if (!ingestJob || ["completed", "failed"].includes(ingestJob.status)) return;
    const t = setInterval(async () => {
      const r = await fetch(`${API}/ingest/status/${ingestJob.job_id}`);
      const d = await r.json();
      setIngestJob(d);
      if (["completed", "failed"].includes(d.status)) clearInterval(t);
    }, 2000);
    return () => clearInterval(t);
  }, [ingestJob]);

  async function refreshConversations() {
    try {
      const r = await fetch(`${API}/conversations`);
      const d = await r.json();
      setConversations(d.conversations || []);
    } catch {}
  }

  async function refreshDocs() {
    try {
      const r = await fetch(`${API}/documents`);
      const d = await r.json();
      setDocs(d.documents || []);
    } catch {}
  }

  async function loadConversation(id) {
    const r = await fetch(`${API}/conversations/${id}`);
    if (!r.ok) return;
    const d = await r.json();
    setConvId(id);
    setMessages(d.messages || []);
  }

  function newChat() {
    setConvId(null);
    setMessages([]);
  }

  async function deleteConversation(id, e) {
    e.stopPropagation();
    await fetch(`${API}/conversations/${id}`, { method: "DELETE" });
    if (id === convId) newChat();
    refreshConversations();
  }

  async function handleUpload(e) {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    setUploading(true);
    const fd = new FormData();
    files.forEach(f => fd.append("files", f));
    try {
      const r = await fetch(`${API}/upload`, { method: "POST", body: fd });
      const d = await r.json();
      await refreshDocs();
      await startIngest("full", d.files);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function deleteDoc(name) {
    await fetch(`${API}/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
    refreshDocs();
  }

  async function startIngest(mode, filenames) {
    const r = await fetch(`${API}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        embedder: settings.embedder, mode, max_pages: 10, language: "en-IN",
        chunk_size: settings.chunk_size, chunk_overlap: settings.chunk_overlap,
        filenames: filenames || null,
      }),
    });
    const d = await r.json();
    if (r.ok) setIngestJob({ job_id: d.job_id, status: "queued" });
    else alert(d.detail || "Ingest failed");
  }

  async function sendMessage(overrideText) {
    const q = (overrideText ?? input).trim();
    if (!q || loading) return;
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setMessages(m => [...m, { role: "user", content: q }]);
    setLoading(true);
    try {
      const r = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q, conversation_id: convId,
          embedder: settings.embedder, reranker: settings.reranker,
          max_loops: settings.max_loops, top_k_retrieve: settings.top_k_retrieve,
          top_n_rerank: settings.top_n_rerank,
        }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "Query failed");
      setConvId(d.conversation_id);
      setMessages(m => [...m, {
        role: "assistant", content: d.answer,
        metadata: { time: d.time_seconds, loops: d.loops, retrieved: d.retrieved, ranked: d.ranked, sources: d.sources, images: d.images },
      }]);
      refreshConversations();
    } catch (err) {
      setMessages(m => [...m, { role: "assistant", content: "⚠️ " + err.message, error: true }]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function autoGrow(e) {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
  }

  return (
    <div className="app">
      <div className="sidebar">
        <div className="sidebar-header">
          <button className="new-chat-btn" onClick={newChat}>
            <Plus size={16} /> New chat
          </button>
        </div>

        <div className="chat-list">
          <div className="chat-list-label">Chats</div>
          {conversations.map(id => (
            <div key={id} className={`chat-item ${convId === id ? "active" : ""}`} onClick={() => loadConversation(id)}>
              <span className="chat-item-title">
                <MessageSquare size={13} style={{ marginRight: 8, verticalAlign: -2, opacity: 0.7 }} />
                {id}
              </span>
              <button className="chat-item-delete" onClick={e => deleteConversation(id, e)}>
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <button className="settings-btn" onClick={() => setShowSettings(true)}>
            <Settings size={16} /> Settings & documents
          </button>
        </div>
      </div>

      <div className="main">
        <div className="main-bg" />
        <div className="messages">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-title">What do you want to know?</div>
              <div className="empty-sub">Ask a question about your ingested documents.</div>
            </div>
          ) : (
            <div className="thread">
              {messages.map((m, i) => (
                <div key={i} className={`msg-row ${m.role}`}>
                  {m.role === "user" ? (
                    <div className="bubble-user">{m.content}</div>
                  ) : (
                    <div className="bubble-assistant">
                      <div className="avatar">AI</div>
                      <div className="assistant-content">
                        <div className={`assistant-text ${m.error ? "error" : ""}`}>{m.content}</div>
                        {m.metadata && (
                          <div className="meta-row">
                            <span className="meta-pill">{m.metadata.time}s</span>
                            <span className="meta-pill">{m.metadata.loops} loops</span>
                            <span className="meta-pill">{m.metadata.retrieved}→{m.metadata.ranked} docs</span>
                            {m.metadata.sources?.map(s => (
                              <span key={s} className="source-pill">{s}</span>
                            ))}
                            {m.metadata.images?.length > 0 && (
                              <div className="meta-images">
                                {m.metadata.images.map(src => (
                                  <img key={src} src={`${API}${src}`} alt="" />
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}
              {loading && (
                <div className="msg-row assistant">
                  <div className="bubble-assistant">
                    <div className="avatar">AI</div>
                    <div className="typing"><span /><span /><span /></div>
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <div className="input-bar">
          <div className="input-inner">
            <div className="input-shell">
              <button className="icon-btn" title="Documents & settings" onClick={() => setShowSettings(true)}>
                <Paperclip size={18} />
              </button>
              <textarea
                ref={textareaRef}
                className="input-textarea"
                rows={1}
                value={input}
                onChange={autoGrow}
                onKeyDown={handleKeyDown}
                placeholder="Message the RAG assistant…"
              />
              <button className="send-btn" disabled={loading || !input.trim()} onClick={() => sendMessage()}>
                <Send size={16} />
              </button>
            </div>
            <div className="input-footer">{settings.embedder} embedder · {settings.reranker} reranker</div>
          </div>
        </div>
      </div>

      {showSettings && (
        <div className="modal-overlay" onClick={() => setShowSettings(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title">Documents & settings</div>
              <button className="modal-close" onClick={() => setShowSettings(false)}><X size={18} /></button>
            </div>

            <div className="modal-section">
              <div className="doc-upload-row">
                <span className="modal-section-label" style={{ marginBottom: 0 }}>Documents</span>
                <button className="upload-btn" onClick={() => fileRef.current?.click()}>
                  {uploading ? <Loader2 size={12} className="spin" /> : <Plus size={12} />} Upload PDF
                </button>
                <input ref={fileRef} type="file" accept=".pdf" multiple hidden onChange={handleUpload} />
              </div>
              <div className="doc-list">
                {docs.length === 0 && <span className="empty-hint">No PDFs uploaded</span>}
                {docs.map(d => (
                  <div key={d} className="doc-item">
                    <FileText size={12} style={{ flexShrink: 0, opacity: 0.7 }} />
                    <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>{d}</span>
                    <button
                      onClick={() => deleteDoc(d)}
                      style={{ background: "none", border: "none", color: "#8a8a92", cursor: "pointer", flexShrink: 0 }}
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                ))}
              </div>
            </div>

            <div className="modal-section">
              <span className="modal-section-label">Chunking</span>
              <div className="field">
                <label className="field-label">Chunk size: {settings.chunk_size}</label>
                <input type="range" min={100} max={4000} step={100} value={settings.chunk_size}
                  onChange={e => setSettings(s => ({ ...s, chunk_size: +e.target.value }))} />
              </div>
              <div className="field" style={{ marginBottom: 0 }}>
                <label className="field-label">Chunk overlap: {settings.chunk_overlap}</label>
                <input type="range" min={0} max={1000} step={50} value={settings.chunk_overlap}
                  onChange={e => setSettings(s => ({ ...s, chunk_overlap: +e.target.value }))} />
              </div>
            </div>

            <div className="modal-section">
              <span className="modal-section-label">Ingest</span>
              <div className="ingest-btns">
                <button className="ingest-btn primary" onClick={() => startIngest("full")}>Full (OCR)</button>
                <button className="ingest-btn secondary" onClick={() => startIngest("index")}>Re-index</button>
              </div>
              {ingestJob && (
                <div className="job-status">
                  <div className="job-status-row">
                    {["queued", "running"].includes(ingestJob.status) && <Loader2 size={12} className="spin" />}
                    <span className={
                      ingestJob.status === "completed" ? "status-completed" :
                      ingestJob.status === "failed" ? "status-failed" : "status-pending"
                    }>{ingestJob.status}</span>
                  </div>
                  {ingestJob.result?.status === "complete" && (
                    <div className="job-detail">
                      {ingestJob.result.text_chunks_added ?? ingestJob.result.text_chunks} text ·{" "}
                      {ingestJob.result.images_added ?? ingestJob.result.images} img ·{" "}
                      {ingestJob.result.tables_added ?? ingestJob.result.tables} tbl
                    </div>
                  )}
                  {ingestJob.result?.detail && ingestJob.status === "failed" && (
                    <div className="job-error">{ingestJob.result.detail}</div>
                  )}
                </div>
              )}
            </div>

            {config && (
              <div className="modal-section" style={{ marginBottom: 0 }}>
                <span className="modal-section-label">Query settings</span>
                <div className="field">
                  <label className="field-label">Embedder</label>
                  <select value={settings.embedder} onChange={e => setSettings(s => ({ ...s, embedder: e.target.value }))}>
                    {Object.entries(config.embedders).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
                  </select>
                </div>
                <div className="field">
                  <label className="field-label">Reranker</label>
                  <select value={settings.reranker} onChange={e => setSettings(s => ({ ...s, reranker: e.target.value }))}>
                    {Object.entries(config.rerankers).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
                  </select>
                </div>
                <div className="field">
                  <label className="field-label">Max agent loops: {settings.max_loops}</label>
                  <input type="range" min={1} max={15} value={settings.max_loops}
                    onChange={e => setSettings(s => ({ ...s, max_loops: +e.target.value }))} />
                </div>
                <div className="field">
                  <label className="field-label">Top-K retrieve: {settings.top_k_retrieve}</label>
                  <input type="range" min={1} max={50} value={settings.top_k_retrieve}
                    onChange={e => setSettings(s => ({ ...s, top_k_retrieve: +e.target.value }))} />
                </div>
                <div className="field" style={{ marginBottom: 0 }}>
                  <label className="field-label">Top-N rerank: {settings.top_n_rerank}</label>
                  <input type="range" min={1} max={20} value={settings.top_n_rerank}
                    onChange={e => setSettings(s => ({ ...s, top_n_rerank: +e.target.value }))} />
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}