import { useEffect, useRef, useState } from "react";
import { Bot, Menu, MessageSquarePlus, Pencil, Send, Trash2, X } from "lucide-react";
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

  async function remove(chat) {
    if (!window.confirm(`「${chat.title}」を削除しますか？`)) return;
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
        const result = await api.send(id, content);
        setMessages(prev => [...prev.filter(item => !item.local), result.user, result.assistant]);
        await refreshChats();
      } catch (e) { setMessages(prev => prev.filter(item => !item.local)); setError(e.message); }
    } else {
      const history = temporary.filter(item => !item.local || item !== welcome).map(({ role, content: body }) => ({ role, content: body }));
      setTemporary(prev => [...prev, { role: "user", content }]);
      try { const result = await api.sendTemporary(content, history); setTemporary(prev => [...prev, result]); }
      catch (e) { setError(e.message); }
    }
    setLoading(false);
  }

  const visible = saved ? messages : temporary;
  const activeTitle = saved ? (chats.find(c => c.id === activeId)?.title || "新しいチャット") : "保存しない会話";

  return <div className="app">
    <aside className={sidebar ? "sidebar open" : "sidebar"}>
      <div className="brand"><span className="bear">K</span><div><strong>KumaGPT</strong><small>ローカル AI</small></div><button className="mobile-close" onClick={() => setSidebar(false)}><X/></button></div>
      <button className="new-chat" onClick={createChat}><MessageSquarePlus size={19}/>新しい保存チャット</button>
      <button className={`temporary ${!saved ? "selected" : ""}`} onClick={startTemporary}>保存しない会話</button>
      <p className="section-label">保存済みの会話</p>
      <div className="chat-list">{chats.map(chat => <div key={chat.id} className={`chat-row ${saved && activeId === chat.id ? "selected" : ""}`}>
        <button className="chat-title" onClick={() => selectChat(chat.id)}>{chat.title}</button>
        <button title="名前を変更" onClick={() => rename(chat)}><Pencil size={15}/></button>
        <button title="削除" onClick={() => remove(chat)}><Trash2 size={15}/></button>
      </div>)}</div>
      <div className="privacy">会話データはこのPC内に保存されます。</div>
    </aside>
    {sidebar && <div className="scrim" onClick={() => setSidebar(false)}/>}
    <main>
      <header><button className="menu" onClick={() => setSidebar(true)}><Menu/></button><div><h1>{activeTitle}</h1><span className={saved ? "mode saved" : "mode"}>{saved ? "保存中" : "一時モード"}</span></div></header>
      <section className="messages">{visible.map((message, index) => <div className={`message ${message.role}`} key={message.id || index}>
        <div className="avatar">{message.role === "assistant" ? <Bot size={20}/> : "私"}</div><div className="bubble">{message.content}</div>
      </div>)}{loading && <div className="message assistant"><div className="avatar"><Bot size={20}/></div><div className="bubble typing"><i/><i/><i/></div></div>}<div ref={endRef}/></section>
      <footer>{error && <div className="error">{error}</div>}<form onSubmit={submit}><textarea value={text} onChange={e => setText(e.target.value)} maxLength={4000} placeholder="KumaGPTにメッセージを送る" rows="1" onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(e); } }}/><button disabled={loading || !text.trim()}><Send size={20}/></button></form><small>Enterで送信・Shift + Enterで改行</small></footer>
    </main>
  </div>;
}
