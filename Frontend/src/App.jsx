import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Trash2, Moon, User, Bot } from 'lucide-react';
import ReactMarkdown from 'react-markdown'


export default function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      content: "Hello! I'm  MemoMate. How can I assist you today?",
      sender: "bot",
      timestamp: new Date()
    }
  ]);
  
  const [inputMessage, setInputMessage] = useState("");
  const messagesEndRef = useRef(null);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  const handleSendMessage = async () => {
    if (inputMessage.trim() === "") return;
    
    // Add user message
    const newUserMessage = {
      id: messages.length + 1,
      content: inputMessage,
      sender: "user",
      timestamp: new Date()
    };
    
    setMessages([...messages, newUserMessage]);
    setInputMessage("");
  
    const response = await fetch("http://localhost:3000/chat", {
      method: "POST",
      headers: {
        "Accept": "application/json",
        "content-type": "application/json"
      },
      body: JSON.stringify({
        prompt: newUserMessage.content
      })
    })

    let botResponse;
    if (!response.ok) {
      botResponse = {
        id: messages.length + 2,
        content: "Failed to process request. Please try again later.",
        sender: "bot",
        timestamp: new Date()
      };
      throw new Error("Failed to fetch data");
    }

    else{
      const data = await response.json();
      botResponse = {
        id: messages.length + 2,
        content: data.reply.reply,
        sender: "bot",
        timestamp: new Date()
      };
    }
      
    setMessages(prevMessages => [...prevMessages, botResponse]);

  };
  
  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  const clearChat = async () => {
    const response = await fetch("http://127.0.0.1:8000/chat-clear",{
      method: "GET",
      headers: {
        "Accept": "application/json"
      }
    })

    if (!response.ok){
      setMessages([{
        id: 1,
        content: "Failed to clear chat. Please try again later",
        sender: "bot",
        timestamp: new Date()
      }]);
      throw new Error("Failed to clear chat");
    }

    setMessages([{
      id: 1,
      content: "Chat cleared. How can I help you today?",
      sender: "bot",
      timestamp: new Date()
    }]);
    console.log("chat cleared")
  };
  
  return (
    <div className="flex flex-col h-screen bg-gray-900 text-gray-100">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-4 flex items-center justify-between">
        <div className="flex-1"></div>
        <div className="flex-1 flex flex-col items-center justify-center">
          <h1 className="font-bold text-s sm:text-xl text-purple-400">MemoMate</h1>
          <div className="flex items-center text-xs text-green-400">
            <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
            Online
          </div>
        </div>
        <div className="flex-1 flex justify-end">
          <button 
            className="p-2 rounded-full hover:bg-gray-700 transition"
            onClick={clearChat}
          >
            <Trash2 size={20} className="text-gray-400 cursor-pointer" />
          </button>
          
        </div>
      </header>
      
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 bg-gray-900">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.map((message) => (
            <div 
              key={message.id} 
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`
                flex max-w-xxl rounded-lg p-4 
                ${message.sender === 'user' 
                  ? 'bg-purple-900 ml-12' 
                  : 'bg-gray-800 mr-12'
                }
              `}>
                <div className="flex-shrink-0 mr-3">
                  {message.sender === 'user' 
                    ? <User size={24} className="text-purple-300" /> 
                    : <Bot size={24} className="text-purple-400" />
                  }
                </div>
                <div className="flex-1">
                  <div className="mb-1 text-sm">
                    {message.sender === 'user' ? 'You' : 'MemoMate'}
                    <span className="ml-2 text-xs text-gray-400">
                      {formatTime(message.timestamp)}
                    </span>
                  </div>
                  <div className="text-sm whitespace-pre-wrap"><ReactMarkdown>{message.content}</ReactMarkdown></div>
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>
      
      {/* Input Area */}
      <div className="bg-gray-800 border-t border-gray-700 p-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-end bg-gray-700 rounded-lg p-2">

            <textarea
              className="flex-1 bg-transparent border-0 focus:ring-0 outline-none resize-none mx-2 text-gray-100 placeholder-gray-400 py-2"
              placeholder="Type your message here..."
              rows={1}
              value={inputMessage}
              onChange={e => setInputMessage(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
            />
            <button 
              className={`p-2 rounded-full ${
                inputMessage.trim() 
                  ? 'bg-purple-600 hover:bg-purple-700' 
                  : 'bg-gray-600 text-gray-400'
              } transition`}
              onClick={handleSendMessage}
              disabled={!inputMessage.trim()}
            >
              <Send size={20} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}