const chatBox = document.getElementById("chatBox");
const session_id = "user1";

function addMessage(text, sender) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender);
    chatBox.appendChild(messageDiv);

    let index = 0;
    function typeLetter() {
        if (index < text.length) {
            messageDiv.textContent += text[index];
            index++;
            setTimeout(typeLetter, 10);
        }
    }
    typeLetter();

    chatBox.scrollTop = chatBox.scrollHeight;
}

function showLoader() {
    const loaderDiv = document.createElement("div");
    loaderDiv.classList.add("loader");
    loaderDiv.id = "loader";
    chatBox.appendChild(loaderDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function removeLoader() {
    const loader = document.getElementById("loader");
    if (loader) loader.remove();
}

async function sendMessage() {
    const userInput = document.getElementById("userInput");
    const message = userInput.value.trim();
    if (!message) return;

    addMessage(message, "user");
    userInput.value = "";

    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", "bot");

    const loader = document.createElement("div");
    loader.classList.add("loader");
    messageDiv.appendChild(loader);
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch("http://172.20.10.5:3000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id, question: message })
        });

        const reader = response.body.getReader();
        let botMessage = "";
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            botMessage += decoder.decode(value, { stream: true });

            if (loader.parentNode) {
                loader.remove();
            }

            messageDiv.innerHTML = botMessage.replace(/\n/g, "<br>");
        }

        chatBox.scrollTop = chatBox.scrollHeight;
    } catch (error) {
        if (loader.parentNode) loader.remove();
        addMessage("Error connecting to server.", "bot");
    }
}
userInput.addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
        event.preventDefault();
        sendMessage();
    }
});