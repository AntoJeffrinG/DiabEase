const familyData = {
    "Alice": { bloodSugar: 150, insulin: 25, exercise: 30, food: 45 },
    "Bob": { bloodSugar: 140, insulin: 20, exercise: 40, food: 50 },
    "Charlie": { bloodSugar: 180, insulin: 30, exercise: 20, food: 60 }
};

function updateData() {
    let selectedMember = document.getElementById("familyMembers").value;
    let data = familyData[selectedMember];

    document.getElementById("memberName").innerText = `${selectedMember}'s Data`;
    document.getElementById("bloodSugar").innerText = data.bloodSugar;
    document.getElementById("insulin").innerText = data.insulin;
    document.getElementById("exercise").innerText = data.exercise;
    document.getElementById("food").innerText = data.food;
}

function addFamilyMember() {
    let name = document.getElementById("newMemberName").value;
    let role = document.getElementById("newMemberRole").value;

    if (name && role) {
        let select = document.getElementById("familyMembers");
        let option = document.createElement("option");
        option.value = name;
        option.textContent = `${name} (${role})`;
        select.appendChild(option);

        // Initialize new member data
        familyData[name] = { bloodSugar: 0, insulin: 0, exercise: 0, food: 0 };

        alert(`${name} added to the family pack!`);
    } else {
        alert("Please enter valid name and role!");
    }
}