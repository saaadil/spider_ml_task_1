// function that runs when we hit the send button
async function sendQuery() {
    // grab the text box and the chat area from the html
    const input = document.getElementById('queryInput');
    const chatBox = document.getElementById('chatBox');
    
    // get what the user typed and remove extra spaces
    const question = input.value.trim();
    
    // if they didn't type anything, just stop and do nothing
    if (!question) return;

    // print the user's question to the screen
    chatBox.innerHTML += `<div class="message"><span class="user">You:</span> ${question}</div>`;
    
    // clear the text box for the next question
    input.value = '';
    
    // force the chat to scroll to the very bottom
    chatBox.scrollTop = chatBox.scrollHeight;

    // make a random id so we can delete this exact loading message later
    const loadingId = 'loading-' + Date.now();
    
    // show a thinking message while we wait for the bot
    chatBox.innerHTML += `<div class="message" id="${loadingId}"><span class="bot">Gemma:</span> Thinking...</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        // send the question to our python backend server
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ question: question })
        });
        
        // wait for the backend to reply with the answer
        const data = await response.json();
        
        // delete the "Thinking..." text
        document.getElementById(loadingId).remove();

        // if the python code broke, show an error message
        if (data.error) {
            chatBox.innerHTML += `<div class="message"><span class="bot">Error:</span> ${data.error}</div>`;
        } else {
            // start building the html for the sources list
            let sourceHtml = '<div class="sources"><strong>Cited Sources:</strong><br>';
            
            // loop through every paper the bot used and add it to the list
            data.sources.forEach(s => {
                sourceHtml += `- ${s.paper} (Match Score: ${s.score})<br>`;
            });
            sourceHtml += '</div>';

            // print the final answer and the sources to the screen
            chatBox.innerHTML += `
                <div class="message">
                    <span class="bot">Gemma:</span> ${data.answer.replace(/\n/g, '<br>')}
                    ${data.sources.length ? sourceHtml : ''}
                </div>
            `;
        }
    } catch (err) {
        // if the server is offline or crashed completely
        document.getElementById(loadingId).remove();
        chatBox.innerHTML += `<div class="message"><span class="bot">Error:</span> Failed to connect to server.</div>`;
    }
    
    // scroll to the bottom one last time to show the new answer
    chatBox.scrollTop = chatBox.scrollHeight;
}