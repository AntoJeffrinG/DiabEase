const chatBox = document.getElementById("chatBox");
const session_id = "user1"; // Unique session ID

function addMessage(text, sender) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender);
    chatBox.appendChild(messageDiv);

    // Display text letter by letter
    let index = 0;
    function typeLetter() {
        if (index < text.length) {
            messageDiv.textContent += text[index];
            index++;
            setTimeout(typeLetter, 10); // Faster response (adjust for speed)
        }
    }
    typeLetter();

    chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll
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

    addMessage(message, "user"); // Display user message
    userInput.value = ""; // Clear input field

    // Create bot response container
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", "bot");

    // Create loader and add it inside bot message container
    const loader = document.createElement("div");
    loader.classList.add("loader");
    messageDiv.appendChild(loader);
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll

    try {
        const response = await fetch("http://192.168.163.64:3000/chat", { // Update to your server IP
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

            // Remove the loader once response starts appearing
            if (loader.parentNode) {
                loader.remove();
            }
            
            // Preserve formatting: new lines and lists
            messageDiv.innerHTML = botMessage.replace(/\n/g, "<br>");
        }

        chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll
    } catch (error) {
        if (loader.parentNode) loader.remove(); // Ensure loader is removed on error
        addMessage("Error connecting to server.", "bot");
    }
}
userInput.addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
        event.preventDefault(); // Prevent default newline behavior
        sendMessage();
    }
});