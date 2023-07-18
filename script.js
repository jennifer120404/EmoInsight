function addMessage(role, content) {
    const chatLog = document.getElementById("chat-log");
    const messageElement = document.createElement("div");
    messageElement.classList.add("message");
    messageElement.classList.add(role);
    messageElement.textContent = content;
    chatLog.appendChild(messageElement);
}

function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    addMessage("user", userInput);

    axios.post("/api/chat", { user_input: userInput })
        .then(response => {
            const reply = response.data.reply;
            addMessage("assistant", reply);
        })
        .catch(error => {
            console.error(error);
        });

    document.getElementById("user-input").value = "";
}