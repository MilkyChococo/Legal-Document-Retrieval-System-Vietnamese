import { useState, useEffect, useRef } from "react";
import bg from "../assets/images/bg.png";
import arrow from "../assets/images/arrow.png";

const Chat = () => {
  const [userInput, setUserInput] = useState("");
  const [isChat, setIsChat] = useState(false);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    const fetchWelcomeMessage = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/");
        const text = await response.text();
        const welcome = {
          sender: "bot",
          text: text || "🧾 Chào bạn! Tôi là trợ lý luật thuế.",
        };
        setMessages([welcome]);
        setIsChat(true); // 👈 Kích hoạt giao diện chat ngay sau khi nhận chào
      } catch (err) {
        console.error("Không thể kết nối đến API:", err);
        setMessages([
          {
            sender: "bot",
            text: "❌ Không thể kết nối đến hệ thống. Vui lòng kiểm tra lại server.",
          },
        ]);
      }
    };

    fetchWelcomeMessage();
  }, []);

  const handleSendMessage = async () => {
    if (userInput.trim() === "") return;

    const trimmedQuestion = userInput.trim();

    // 1. Thêm câu hỏi vào khung chat
    const userMessage = {
      sender: "user",
      text: trimmedQuestion,
    };
    setMessages((prev) => [...prev, userMessage]);
    setUserInput("");
    setLoading(true);

    try {
      // 2. Gửi câu hỏi lên Flask
      const response = await fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmedQuestion }),
      });

      const data = await response.json();

      // 3. Kiểm tra và thêm câu trả lời từ bot
      const botMessage = {
        sender: "bot",
        text: data.answer,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error("❌ Lỗi khi gọi API:", err);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "❌ Lỗi kết nối đến server.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Tự động scroll xuống cuối khi có tin nhắn mới
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div
      className="w-full h-full flex justify-center items-center"
      style={{ padding: "0px" }}
    >
      {isChat ? (
        <div className="w-[60%] h-[85%] bg-[#f2f2f2] rounded-xl px-12 py-8 flex flex-col justify-between items-center">
          <div
            className="w-[90%] px-10 overflow-y-auto custom-scrollbar-navbar flex flex-col space-y-2"
            id="messages"
            style={{ height: "80%" }}
          >
            {messages.map((message, index) => (
              <div
                key={index}
                className={`py-2 px-4 rounded-lg max-w-[70%] ${
                  message.sender === "user"
                    ? "bg-blue-200 text-right self-end text-black"
                    : "bg-gray-200 text-left self-start text-black"
                }`}
              >
                {message.text}
              </div>
            ))}
            {loading && (
              <div className="text-sm italic text-gray-500 self-start">
                Đang trả lời...
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="w-[90%] h-[20%] flex flex-row space-x-1 items-end rounded-2xl shadow-lg border border-gray-300 p-2">
            <textarea
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              id="user-input"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              placeholder="Hỏi bất kỳ điều gì"
              className="font-normal text-[14px] text-primary-500 h-full w-full px-4 rounded-l-2xl bg-primary-600 no-focus focus:outline-none focus:ring-0 overflow-auto resize-none custom-scrollbar-navbar"
            />
            <button
              id="send-btn"
              onClick={handleSendMessage}
              className="w-[30px] h-[30px] rounded-full bg-primary-400 aspect-square cursor-pointer"
            >
              <img src={arrow} alt="Gửi" />
            </button>
          </div>
        </div>
      ) : (
        <div
          className="w-[60%] h-[80%] bg-[#f2f2f2] rounded-xl flex flex-col justify-end items-center p-12 space-y-10"
          style={{
            backgroundImage: `url(${bg})`,
            backgroundSize: "contain",
            backgroundPosition: "center",
            backgroundRepeat: "no-repeat",
          }}
        >
          <h1 className="text-2xl text-[#db0000] font-semibold text-center">
            Bạn có câu hỏi gì về luật thuế?
          </h1>
          <div className="w-[80%] h-[20%] flex flex-row space-x-1 items-end rounded-2xl shadow-lg border border-gray-300 p-2">
            <textarea
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              id="user-input"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              placeholder="Hỏi bất kỳ điều gì"
              className="font-normal text-[14px] text-primary-500 h-full w-full px-4 rounded-l-2xl bg-primary-600 no-focus focus:outline-none focus:ring-0 overflow-auto resize-none custom-scrollbar-navbar"
            />
            <button
              id="send-btn"
              onClick={handleSendMessage}
              className="w-[30px] h-[30px] rounded-full bg-primary-400 aspect-square cursor-pointer"
            >
              <img src={arrow} alt="Gửi" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Chat;
