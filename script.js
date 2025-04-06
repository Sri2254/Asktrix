let selectedLanguage = 'en-US';
let speechSynthesisLang = 'en-US';
let isReadAloudEnabled = true;
let currentTheme = 'light';

// Theme switching functionality
function toggleTheme() {
    const htmlElement = document.documentElement;
    const newTheme = htmlElement.getAttribute('data-bs-theme') === 'dark' ? 'light' : 'dark';
    htmlElement.setAttribute('data-bs-theme', newTheme);
    currentTheme = newTheme;
    
    // Save theme preference to server
    fetch('/api/theme', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ theme: newTheme }),
    })
    .catch(error => console.error('Error saving theme preference:', error));
}

// Load saved theme
function loadThemePreference() {
    fetch('/api/theme')
        .then(response => response.json())
        .then(data => {
            if (data.theme) {
                document.documentElement.setAttribute('data-bs-theme', data.theme);
                currentTheme = data.theme;
                document.getElementById('themeSwitch').checked = data.theme === 'dark';
            }
        })
        .catch(error => console.error('Error loading theme preference:', error));
}

function loadConversations() {
    fetch('/api/conversations')
        .then(response => response.json())
        .then(data => {
            const historyList = document.getElementById('conversation-history');
            if (historyList) {
                historyList.innerHTML = '';
                data.forEach(conv => {
                    const li = document.createElement('li');
                    li.className = 'conversation-item';
                    li.innerHTML = `<a href="#" data-id="${conv.id}">${conv.title}</a>`;
                    historyList.appendChild(li);
                });
            }
        })
        .catch(error => console.error('Error loading conversations:', error));
}

document.addEventListener('DOMContentLoaded', function () {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const exportBtn = document.getElementById('export-chat-btn');
    const readAloudSwitch = document.getElementById('readAloudSwitch');
    const voiceInputBtn = document.getElementById('voice-input-btn');
    const languageSelect = document.getElementById('language-select');
    const themeSwitch = document.getElementById('themeSwitch');
    
    // Initialize theme
    loadThemePreference();
    
    // Setup theme toggle event
    if (themeSwitch) {
        themeSwitch.addEventListener('change', toggleTheme);
    }
    
    loadConversations();
    
    // Error handling for chat form
    if (!chatForm) {
        console.error('Chat form not found!');
        return;
    }

    // Add click handler for the send button as a fallback
    const sendButton = document.getElementById('send-button');
    if (sendButton) {
        sendButton.addEventListener('click', function (e) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        });
    }

    let recognition;

    // Fetch and display chat history
    fetch('/api/chat/history')
        .then(response => response.json())
        .then(data => {
            let lastConversationId = null;
            data.reverse().forEach(chat => {
                const timestamp = chat.timestamp || Date.now();

                if (chat.conversation_id && chat.conversation_id !== lastConversationId) {
                    const convoHeader = document.createElement('div');
                    convoHeader.className = 'text-center text-info fw-bold mt-4';
                    convoHeader.textContent = `🗂 Conversation: #${chat.conversation_id}`;
                    chatMessages.appendChild(convoHeader);
                    lastConversationId = chat.conversation_id;
                }

                addMessageToChat(chat.message, chat.response, chat.is_financial, timestamp);
            });
        })
        .catch(error => {
            console.error('Error fetching chat history:', error);
            const errorDiv = document.createElement('div');
            errorDiv.className = 'alert alert-danger';
            errorDiv.textContent = 'Failed to load chat history. Please refresh the page.';
            chatMessages.appendChild(errorDiv);
        });

    // Handle form submit
    if (chatForm) {
        chatForm.addEventListener('submit', function (e) {
            e.preventDefault();
            const message = chatInput.value.trim();
            if (message) {
                const timestamp = Date.now();
                addMessageToChat(message, '...', false, timestamp);
                
                // Add loading indicator
                const loadingIndicator = document.createElement('div');
                loadingIndicator.id = 'loading-indicator';
                loadingIndicator.className = 'text-center my-2';
                loadingIndicator.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>';
                chatMessages.appendChild(loadingIndicator);
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                // Replace the existing fetch call with your new version
                fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, language: selectedLanguage })
                })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(err => {
                            throw new Error(err.error || 'Request failed');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    // Remove loading indicator
                    const loadingElement = document.getElementById('loading-indicator');
                    if (loadingElement) loadingElement.remove();
                    
                    // Update chat UI
                    const botMessages = document.querySelectorAll('.bot-message');
                    const lastBotMessage = botMessages[botMessages.length - 1];
                    if (lastBotMessage) {
                        const messageContent = lastBotMessage.querySelector('.message-content');
                        if (messageContent) {
                            const timeString = new Date(timestamp).toLocaleTimeString();
                            messageContent.innerHTML = `<strong>Bot:</strong> ${data.reply}<br><small>${timeString}</small>`;
                        }
                    }
                    
                    if (isReadAloudEnabled) {
                        speakText(data.reply);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    // Remove loading indicator
                    const loadingElement = document.getElementById('loading-indicator');
                    if (loadingElement) loadingElement.remove();
                    
                    // Show error to user
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'alert alert-danger';
                    errorDiv.textContent = error.message;
                    document.getElementById('chat-messages').appendChild(errorDiv);
                });

                chatInput.value = '';
                chatInput.focus();
            }
        });
    }

    // Voice input
    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;

        recognition.onresult = function (event) {
            chatInput.value = event.results[0][0].transcript;
        };

        recognition.onerror = function (event) {
            console.error('Speech recognition error:', event.error);
        };
    }

    if (voiceInputBtn) {
        voiceInputBtn.addEventListener('click', function () {
            if (recognition) {
                recognition.lang = selectedLanguage;
                recognition.start();
            } else {
                alert('Speech recognition not supported in this browser.');
            }
        });
    }

    // Language selection
    if (languageSelect) {
        languageSelect.addEventListener('change', function () {
            selectedLanguage = this.value;
            speechSynthesisLang = selectedLanguage;
        });
    }

    // Read aloud switch toggle
    if (readAloudSwitch) {
        readAloudSwitch.addEventListener('change', function () {
            isReadAloudEnabled = this.checked;
        });
    }

    // Export chat
    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            fetch('/api/chat/history')
                .then(response => response.json())
                .then(data => {
                    const exportLines = [];

                    data.reverse().forEach(chat => {
                        const time = new Date(chat.timestamp || Date.now()).toLocaleString();
                        exportLines.push(`[${time}] You: ${chat.message}`);
                        exportLines.push(`[${time}] Bot: ${chat.response}`);
                        exportLines.push('');
                    });

                    const blob = new Blob([exportLines.join('\n')], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);

                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `chat-history-${Date.now()}.txt`;
                    a.click();
                    URL.revokeObjectURL(url);
                })
                .catch(error => {
                    console.error('Error exporting chat:', error);
                    alert('Failed to export chat history. Please try again.');
                });
        });
    }
});

// Function to add messages to the chat UI
function addMessageToChat(userMessage, botResponse, isFinancial, timestamp) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;

    const time = new Date(timestamp).toLocaleTimeString();

    const userDiv = document.createElement('div');
    userDiv.className = 'message user-message';
    userDiv.innerHTML = `<div class="message-content"><strong>You:</strong> ${userMessage}<br><small>${time}</small></div>`;
    chatMessages.appendChild(userDiv);

    const botDiv = document.createElement('div');
    botDiv.className = 'message bot-message';
    botDiv.innerHTML = `<div class="message-content"><strong>Bot:</strong> ${botResponse}<br><small>${time}</small></div>`;
    if (isFinancial) {
        botDiv.classList.add('financial-response');
    }
    chatMessages.appendChild(botDiv);

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Function to read text aloud
function speakText(text) {
    if ('speechSynthesis' in window && isReadAloudEnabled) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = speechSynthesisLang;
        window.speechSynthesis.cancel(); // Cancel any ongoing speech
        window.speechSynthesis.speak(utterance);
    }
}

// Utility function to generate a UUID
function generateConversationId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Utility function to display warning messages
function displayWarning(message) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    
    const warningDiv = document.createElement('div');
    warningDiv.className = 'alert alert-warning';
    warningDiv.textContent = message;
    chatMessages.appendChild(warningDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to show the warning
}