document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const chatMessages = document.getElementById('chat-messages');
    const clearBtn = document.getElementById('clear-btn');
    const suggestions = document.querySelectorAll('.suggestion-chip');

    // State
    let chatHistory = [];
    let isWaitingForResponse = false;

    // Enable/disable send button based on input
    messageInput.addEventListener('input', () => {
        sendBtn.disabled = messageInput.value.trim() === '' || isWaitingForResponse;
    });

    // Handle suggestion clicks
    suggestions.forEach(chip => {
        chip.addEventListener('click', () => {
            const text = chip.textContent;
            messageInput.value = text;
            sendBtn.disabled = false;
            chatForm.dispatchEvent(new Event('submit'));
        });
    });

    clearBtn.addEventListener('click', () => {
        chatHistory = [];
        chatMessages.innerHTML = `
            <div class="message ai-message">
                <div class="message-content">Hey! I'm Christian Agyapong, but most people know me as Chrix Tech. I study Computer Science at the University of Ghana, majoring in AI and Machine Learning. Feel free to ask me anything about my background, projects, or goals!</div>
            </div>
        `;
    });

    // Form submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const messageText = messageInput.value.trim();
        if (!messageText || isWaitingForResponse) return;

        // Add user message to UI
        addMessageToUI('user', messageText);
        
        // Clear input and state
        messageInput.value = '';
        sendBtn.disabled = true;
        isWaitingForResponse = true;

        // Show typing indicator
        const typingId = showTypingIndicator();

        try {
            // Send to API
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: messageText,
                    history: chatHistory
                })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            const aiReply = data.reply;
            const newSuggestions = data.suggestions || [];

            // Remove typing indicator
            removeTypingIndicator(typingId);

            // Add AI response to UI
            addMessageToUI('ai', aiReply);

            // Update history
            chatHistory.push([messageText, aiReply]);

            // Update suggestions UI
            updateSuggestions(newSuggestions);

        } catch (error) {
            console.error('Error:', error);
            removeTypingIndicator(typingId);
            addMessageToUI('ai', 'Oops, something went wrong connecting to the server. Please try again.');
        } finally {
            isWaitingForResponse = false;
            // Focus input if not on mobile
            if (window.innerWidth > 768) {
                messageInput.focus();
            }
        }
    });

    // Helpers
    function escapeHTML(str) {
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag] || tag)
        );
    }

    function urlify(text) {
        const escaped = escapeHTML(text);
        // Match URLs but exclude trailing punctuation like periods or commas
        const urlRegex = /(https?:\/\/[^\s]+[^\s.,;:!?'"()])/g;
        return escaped.replace(urlRegex, function(url) {
            return `<a href="${url}" target="_blank" rel="noopener noreferrer" style="color: #58a6ff; text-decoration: underline;">${url}</a>`;
        });
    }

    function addMessageToUI(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = urlify(text);
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function showTypingIndicator() {
        const id = 'typing-' + Date.now();
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message ai-message';
        typingDiv.id = id;
        
        typingDiv.innerHTML = `
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        
        chatMessages.appendChild(typingDiv);
        scrollToBottom();
        return id;
    }

    function removeTypingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Adjust UI when virtual keyboard opens (mobile)
    function adjustForKeyboard() {
        const chatInputArea = document.querySelector('.chat-input-area');
        if (!chatInputArea) return;

        // Use VisualViewport API when available to compute keyboard height
        if (window.visualViewport) {
            const vv = window.visualViewport;
            const adjust = () => {
                // Distance the viewport height changed from layout viewport
                const keyboardHeight = Math.max(0, window.innerHeight - vv.height);
                // Move input area up and add padding so messages aren't hidden
                chatInputArea.style.transform = `translateY(-${keyboardHeight}px)`;
                chatMessages.style.paddingBottom = `${keyboardHeight + 24}px`;
                // keep latest message visible
                scrollToBottom();
            };

            vv.addEventListener('resize', adjust);
            vv.addEventListener('scroll', adjust);
            // call once to initialize
            adjust();
        } else {
            // Fallback: on focus add extra padding; remove on blur
            messageInput.addEventListener('focus', () => {
                chatMessages.style.paddingBottom = '300px';
                scrollToBottom();
            });
            messageInput.addEventListener('blur', () => {
                chatMessages.style.paddingBottom = '';
                chatInputArea.style.transform = '';
            });
        }
    }

    // Initialize keyboard adjustments
    adjustForKeyboard();

    function updateSuggestions(newSuggestions) {
        const suggestionsContainer = document.getElementById('suggestions');
        suggestionsContainer.innerHTML = ''; // Clear old suggestions
        
        newSuggestions.forEach(text => {
            const chip = document.createElement('button');
            chip.className = 'suggestion-chip';
            chip.textContent = text;
            chip.addEventListener('click', () => {
                messageInput.value = text;
                sendBtn.disabled = false;
                chatForm.dispatchEvent(new Event('submit'));
            });
            suggestionsContainer.appendChild(chip);
        });
    }
});
