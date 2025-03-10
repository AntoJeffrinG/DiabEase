// Sample data for users and their posts
let users = [
  { username: "user1", steps: 10000, streak: 5 },
  { username: "user2", steps: 12000, streak: 7 },
  { username: "user3", steps: 8000, streak: 3 }
];

let messages = [
  { username: "user1", content: "Struggling with my blood sugar today, any tips?" },
  { username: "user2", content: "I reached 5000 steps today, feeling great!" }
];

// Function to log steps for today
function logSteps() {
  const stepsToday = parseInt(prompt("Enter your steps for today:"));
  const user = getUserData(); // Get logged-in user

  // Check if the user reached 5000 steps
  if (stepsToday >= 5000) {
    user.streak++;
  } else {
    user.streak = 0; // reset streak
  }

  user.steps += stepsToday;

  // Update UI elements based on the current page
  if (document.getElementById('profile-steps')) {
    document.getElementById('profile-steps').textContent = user.steps;
    document.getElementById('profile-streak').textContent = user.streak;
  }

  // Refresh leaderboard
  updateLeaderboard();
}

// Get current user data (this is just a placeholder for now)
function getUserData() {
  return users[0]; // This would be dynamic based on the logged-in user
}

// Update the leaderboard
function updateLeaderboard() {
  const tableBody = document.getElementById('leaderboard-table').querySelector('tbody');
  tableBody.innerHTML = ''; // Clear existing leaderboard

  // Sort users by steps walked
  users.sort((a, b) => b.steps - a.steps);

  users.forEach((user, index) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${index + 1}</td>
      <td>${user.username}</td>
      <td>${user.steps}</td>
    `;
    tableBody.appendChild(row);
  });
}

// Post a message to the message board
function postMessage() {
  const messageContent = document.getElementById('message-input').value;
  if (messageContent.trim() !== "") {
    const user = getUserData(); // Get logged-in user
    messages.push({ username: user.username, content: messageContent });

    // Clear the input
    document.getElementById('message-input').value = "";

    // Update message board
    displayMessages();
  }
}

// Display all messages
function displayMessages() {
  const container = document.getElementById('messages-container');
  container.innerHTML = ''; // Clear existing messages

  messages.forEach(message => {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.innerHTML = `<strong>${message.username}:</strong> ${message.content}`;
    container.appendChild(messageDiv);
  });
}

// Initial calls to load data
updateLeaderboard();
displayMessages();