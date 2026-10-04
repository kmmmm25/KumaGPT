import { useEffect, useRef, useState } from "react";
import { Bot, Clock3, Menu, MessageSquarePlus, Pencil, Send, ShieldCheck, SlidersHorizontal, Sparkles, Trash2, X } from "lucide-react";
import { api } from "./api";

const welcome = { role: "assistant", content: "こんにちは。KumaGPTです。今日は何を話しましょう？", local: true };

export default function App() {
  const [saved, setSaved] = useState(true);
  const [chats, setChats] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([welcome]);
  const [temporary, setTemporary] = useState([welcome]);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebar, setSidebar] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(30);
  const endRef = useRef(null);

  const refreshChats = async () => setChats((await api.chats()).items);
  useEffect(() => { refreshChats().catch(e => setError(e.message)); }, []);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, temporary, loading]);

  async function selectChat(id) {
    setError(""); setSaved(true); setActiveId(id); setSidebar(false);
    try { const data = await api.messages(id); setMessages(data.items.length ? data.items : [welcome]); }
    catch (e) { setError(e.message); }
  }

  function startTemporary() { setSaved(false); setActiveId(null); setTemporary([welcome]); setSidebar(false); setError(""); }

  async function createChat() {
    try { const chat = await api.createChat(); await refreshChats(); await selectChat(chat.id); }
    catch (e) { setError(e.message); }
  }

  async function rename(chat) {
    const title = window.prompt("新しいタイトル", chat.title);
    if (!title?.trim()) return;
    try { await api.renameChat(chat.id, title.trim()); await refreshChats(); }
    catch (e) { setError(e.message); }
  }

  function remove(chat) { setDeleteTarget(chat); }

  async function confirmRemove() {
    const chat = deleteTarget;
    if (!chat) return;
    setDeleteTarget(null);
    try {
      await api.deleteChat(chat.id); if (activeId === chat.id) { setActiveId(null); setMessages([welcome]); }
      await refreshChats();
    } catch (e) { setError(e.message); }
  }

  async function submit(event) {
    event.preventDefault();
    const content = text.trim(); if (!content || loading) return;
    setText(""); setError(""); setLoading(true);
    if (saved) {
      try {
        let id = activeId;
        if (!id) { const chat = await api.createChat(); id = chat.id; setActiveId(id); }
        setMessages(prev => [...prev.filter(item => !item.local), { role: "user", content, local: true }]);
        const result = await api.send(id, content, { temperature, top_k: topK });
        setMessages(prev => [...prev.filter(item => !item.local), result.user, result.assistant]);
        await refreshChats();
      } catch (e) { setMessages(prev => prev.filter(item => !item.local)); setError(e.message); }
    } else {
      const history = temporary.filter(item => !item.local || item !== welcome).map(({ role, content: body }) => ({ role, content: body }));
      setTemporary(prev => [...prev, { role: "user", content }]);
      try { const result = await api.sendTemporary(content, history, { temperature, top_k: topK }); setTemporary(prev => [...prev, result]); }
      catch (e) { setError(e.message); }
    }
    setLoading(false);
  }

  const visible = saved ? messages : temporary;
  const activeTitle = saved ? (chats.find(c => c.id === activeId)?.title || "新しいチャット") : "保存しない会話";
  const hasConversation = visible.some(message => message.role === "user");
  const suggestions = ["今日のアイデアを一緒に考えて", "短い物語を書いて", "Pythonについて教えて"];

  return <div className="app">
    <aside className={sidebar ? "sidebar open" : "sidebar"}>
      <div className="brand"><span className="bear"><span>K</span></span><div><strong>KumaGPT</strong><small>PERSONAL INTELLIGENCE</small></div><button className="mobile-close icon-button" onClick={() => setSidebar(false)} aria-label="メニューを閉じる"><X size={18}/></button></div>
      <button className="new-chat" onClick={createChat}><span className="new-icon"><MessageSquarePlus size={17}/></span><span>新しいチャット</span></button>
      <button className={`temporary ${!saved ? "selected" : ""}`} onClick={startTemporary}><Clock3 size={16}/><span>保存しない会話</span></button>
      <p className="section-label">ライブラリ</p>
      <div className="chat-list">{chats.map(chat => <div key={chat.id} className={`chat-row ${saved && activeId === chat.id ? "selected" : ""}`}>
        <button className="chat-title" onClick={() => selectChat(chat.id)}>{chat.title}</button>
        <div className="row-actions"><button title="名前を変更" onClick={() => rename(chat)}><Pencil size={14}/></button>
        <button title="削除" onClick={() => remove(chat)}><Trash2 size={14}/></button></div>
      </div>)}</div>
      <div className="privacy"><span className="privacy-icon"><ShieldCheck size={16}/></span><span><strong>On-device storage</strong><small>会話はこのPC内だけに保存</small></span></div>
    </aside>
    {sidebar && <div className="scrim" onClick={() => setSidebar(false)}/>}
    {deleteTarget && <div className="modal-backdrop" role="presentation" onMouseDown={() => setDeleteTarget(null)}>
      <div className="confirm-dialog" role="alertdialog" aria-modal="true" aria-labelledby="delete-dialog-title" aria-describedby="delete-dialog-description" onMouseDown={event => event.stopPropagation()}>
        <span className="dialog-icon"><Trash2 size={20}/></span>
        <h2 id="delete-dialog-title">チャットを削除しますか？</h2>
        <p id="delete-dialog-description">「{deleteTarget.title}」のメッセージは元に戻せません。</p>
        <div className="dialog-actions"><button onClick={() => setDeleteTarget(null)}>キャンセル</button><button className="danger" onClick={confirmRemove}>削除する</button></div>
      </div>
    </div>}
    <main>
      <header><button className="menu icon-button" onClick={() => setSidebar(true)} aria-label="メニューを開く"><Menu size={20}/></button><div className="header-copy"><h1>{activeTitle}</h1><span className={saved ? "mode saved" : "mode"}><i/>{saved ? "保存中" : "一時セッション"}</span></div><div className="header-actions"><button className={`sampling-button ${settingsOpen ? "active" : ""}`} onClick={() => setSettingsOpen(open => !open)} aria-expanded={settingsOpen} aria-controls="sampling-panel"><SlidersHorizontal size={15}/><span>生成設定</span></button><div className="model-chip"><span className="model-dot"/>KumaGPT 4</div></div></header>
      {settingsOpen && <div className="sampling-panel" id="sampling-panel">
        <div className="sampling-heading"><div><strong>生成のばらつき</strong><small>次のメッセージから反映されます</small></div><button className="sampling-close" onClick={() => setSettingsOpen(false)} aria-label="生成設定を閉じる"><X size={16}/></button></div>
        <label><span><b>Temperature</b><output>{temperature.toFixed(1)}</output></span><input type="range" min="0.1" max="2" step="0.1" value={temperature} onChange={event => setTemperature(Number(event.target.value))}/><small>低いほど安定、高いほど意外な表現になります。</small></label>
        <label><span><b>Top-k</b><output>{topK}</output></span><input type="range" min="1" max="100" step="1" value={topK} onChange={event => setTopK(Number(event.target.value))}/><small>次の単語候補を上位何件まで残すかを指定します。</small></label>
        <button className="sampling-reset" onClick={() => { setTemperature(0.7); setTopK(30); }}>既定値に戻す</button>
      </div>}
      <section className={`messages ${hasConversation ? "" : "empty"}`}>
        {!hasConversation ? <div className="hero">
          <div className="hero-orb"><div className="orb-core"><Sparkles size={32}/></div></div>
          <p className="eyebrow">KUMAGPT 4</p>
          <h2>何を一緒に<br/><span>考えましょう？</span></h2>
          <p className="hero-copy">あなたのPCで動く、静かでプライベートなAI。<br/>アイデアから小さな疑問まで、気軽に話しかけてください。</p>
          <div className="suggestions">{suggestions.map(suggestion => <button key={suggestion} onClick={() => setText(suggestion)}>{suggestion}<span>↗</span></button>)}</div>
        </div> : visible.map((message, index) => <div className={`message ${message.role}`} key={message.id || index}>
          <div className="avatar">{message.role === "assistant" ? <Bot size={17}/> : <span>YOU</span>}</div><div className="message-content"><span className="message-label">{message.role === "assistant" ? "KumaGPT" : "あなた"}</span><div className="bubble">{message.content}</div></div>
        </div>)}
        {loading && <div className="message assistant"><div className="avatar"><Bot size={17}/></div><div className="message-content"><span className="message-label">KumaGPT</span><div className="bubble typing"><i/><i/><i/></div></div></div>}<div ref={endRef}/>
      </section>
      <footer>{error && <div className="error">{error}</div>}<form onSubmit={submit}><textarea value={text} onChange={e => setText(e.target.value)} maxLength={4000} placeholder="メッセージを入力" rows="1" aria-label="メッセージ" onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(e); } }}/><button aria-label="送信" disabled={loading || !text.trim()}><Send size={18}/></button></form><small><span>KumaGPTは誤った情報を生成する場合があります。</span><span>Enterで送信&nbsp;&nbsp;·&nbsp;&nbsp;Shift + Enterで改行</span></small></footer>
    </main>
  </div>;
}
