document.addEventListener('DOMContentLoaded', () => {
    // Check authentication
    const token = localStorage.getItem('token');
    const user = JSON.parse(localStorage.getItem('user'));

    if (!token || !user) {
        window.location.href = '/login.html';
        return;
    }

    // DOM Elements
    const form = document.getElementById('chat-form');
    const input = document.getElementById('chat-input');
    const messages = document.getElementById('messages');
    const modelSelect = document.getElementById('modelSelect');
    const modelDropdown = document.getElementById('modelDropdown');
    const selectedModelText = document.getElementById('selectedModelText');
    const modelOptions = document.querySelectorAll('.model-option');
    const submitBtn = document.getElementById('submit-btn');
    const headerRight = document.querySelector('.chat-header .auth-buttons');

    // Update header with user info
    headerRight.innerHTML = `
        <div class="user-container">
            <div class="user-trigger" id="userTrigger">
                <span>${user.username}</span>
                <i class="fas fa-user-circle"></i>
            </div>
            <div class="user-menu" id="userMenu">
                <div class="menu-item">
                    <i class="fas fa-envelope"></i>
                    <span>${user.email}</span>
                </div>
                <a href="/album.html" class="menu-item">
                    <i class="fas fa-images"></i>
                    <span>My Album</span>
                </a>
                <div class="menu-item" id="logoutButton">
                    <i class="fas fa-sign-out-alt"></i>
                    <span>Logout</span>
                </div>
            </div>
        </div>
    `;

    // Add user menu functionality
    const userTrigger = document.getElementById('userTrigger');
    const userMenu = document.getElementById('userMenu');
    const logoutButton = document.getElementById('logoutButton');

    userTrigger.addEventListener('click', (e) => {
        e.stopPropagation();
        userMenu.classList.toggle('show');
    });

    logoutButton.addEventListener('click', async () => {
        try {
            const response = await fetch('/auth/logout', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            
            if (!response.ok) throw new Error('Logout failed');
            
            localStorage.removeItem('token');
            localStorage.removeItem('user');
            window.location.href = '/login.html';
        } catch (error) {
            console.error('Logout error:', error);
        }
    });

    let selectedModel = 'sdxl';
    let isGenerating = false;

    // Model Selection Menu
    modelSelect?.addEventListener('click', (e) => {
        e.stopPropagation();
        modelDropdown.classList.toggle('show');
    });

    modelOptions?.forEach(option => {
        option.addEventListener('click', (e) => {
            e.stopPropagation();
            selectedModel = option.dataset.model;
            selectedModelText.textContent = option.querySelector('.model-name').textContent;
            modelOptions.forEach(opt => opt.classList.remove('active'));
            option.classList.add('active');
            modelDropdown.classList.remove('show');
        });
    });

    // Close all dropdowns on outside click
    document.addEventListener('click', e => {
        if (!modelDropdown.contains(e.target) && !modelSelect.contains(e.target)) {
            modelDropdown.classList.remove('show');
        }
        if (!userMenu.contains(e.target) && !userTrigger.contains(e.target)) {
            userMenu.classList.remove('show');
        }
    });

    // Handle image generation
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (isGenerating) return;

        const prompt = input.value.trim();
        if (!prompt) return;

        try {
            isGenerating = true;
            submitBtn.disabled = true;
            submitBtn.classList.add('loading');
            
            // Add user message
            addMessage('user', prompt);
            input.value = '';
            input.style.height = 'auto';

            const loadingId = addMessage('ai', '<div class="loader"></div>');

            // Send request to backend
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`, // Make sure token is included
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    prompt,
                    model: selectedModel
                })
            });

            // Update the error handling to show more details
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Image generation failed');
            }

            if (!data.data?.url) {
                throw new Error('No image URL in response');
            }

            // Update how we access the imageUrl from response
            const imageMessage = `
                <div class="generated-image">
                    <img src="${data.data.url}" alt="Generated image" loading="lazy">
                    <p class="prompt-text">${prompt}</p>
                </div>
            `;
            updateMessage(loadingId, imageMessage);

            // Save to localStorage
            saveImageToLocalStorage(data.data.url, prompt);

        } catch (error) {
            console.error("Generation error:", error);
            addMessage('ai', `Error: ${error.message}`);
        } finally {
            isGenerating = false;
            submitBtn.disabled = false;
            submitBtn.classList.remove('loading');
        }
    });

    // Helper Functions
    function addMessage(type, content) {
        const messageId = Date.now();
        const messageHtml = `
            <div class="message ${type}" id="msg-${messageId}">
                <div class="message-content">
                    <div class="avatar">${type === 'ai' ? 'AI' : 'You'}</div>
                    <div class="text">${content}</div>
                </div>
            </div>
        `;
        messages.insertAdjacentHTML('beforeend', messageHtml);
        messages.scrollTop = messages.scrollHeight;
        return messageId;
    }

    function updateMessage(id, content) {
        const message = document.getElementById(`msg-${id}`);
        if (message) {
            message.querySelector('.text').innerHTML = content;
        }
    }

    function saveImageToLocalStorage(imageUrl, prompt) {
        const stored = JSON.parse(localStorage.getItem('generatedImages') || '[]');
        stored.push({ imageUrl, prompt, model: selectedModel });
        localStorage.setItem('generatedImages', JSON.stringify(stored));
    }

    // Auto-grow textarea
    input.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = `${this.scrollHeight}px`;
    });
});