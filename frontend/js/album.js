document.addEventListener('DOMContentLoaded', async () => {
    const token = localStorage.getItem('token');
    const user = JSON.parse(localStorage.getItem('user'));

    if (!token || !user) {
        window.location.href = '/login.html';
        return;
    }

    // Add user info to header
    const header = document.querySelector('.gallery-header');
    if (!header) {
        console.error('Gallery header not found');
        return;
    }

    header.innerHTML = `
        <h1>My Generated Images</h1>
        <div class="user-avatar">
            <i class="fas fa-user-circle"></i>
            <span>${user.username}</span>
            <div class="user-menu" id="userMenu">
                <div class="user-info">
                    <i class="fas fa-envelope"></i>
                    <span>${user.email}</span>
                </div>
                <a href="/" class="menu-item">
                    <i class="fas fa-home"></i>
                    <span>Home</span>
                </a>
                <div class="menu-item" id="logoutButton">
                    <i class="fas fa-sign-out-alt"></i>
                    <span>Logout</span>
                </div>
            </div>
        </div>
    `;

    // Add event listeners after DOM elements are created
    const userAvatar = document.querySelector('.user-avatar');
    const userMenu = document.getElementById('userMenu');
    const logoutButton = document.getElementById('logoutButton');

    // Toggle menu
    userAvatar.addEventListener('click', (e) => {
        e.stopPropagation();
        userMenu.classList.toggle('show');
    });

    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!userAvatar.contains(e.target)) {
            userMenu.classList.remove('show');
        }
    });

    // Attach logout handler
    logoutButton.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        handleLogout();
    });

    const imageGrid = document.querySelector('.image-grid');

    // Load and display images
    async function loadUserImages() {
        try {
            console.log('Fetching images...');
            const response = await fetch('/api/images', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!response.ok) {
                throw new Error('Failed to fetch images');
            }

            const { data: images } = await response.json();
            console.log('Received images:', images);

            // Clear existing content
            imageGrid.innerHTML = '';

            if (!images || images.length === 0) {
                imageGrid.innerHTML = `
                    <div class="no-images">
                        <p>No images yet. Go create some!</p>
                        <a href="/" class="create-btn">Create Images</a>
                    </div>
                `;
                return;
            }

            // Display each image
            images.forEach(image => {
                const imageCard = `
                    <div class="image-card">
                        <div class="image-container">
                            <img src="${image.url}" alt="AI Generated Image" loading="lazy">
                            <div class="image-overlay">
                                <p class="prompt">${image.prompt}</p>
                                <div class="image-actions">
                                    <button onclick="downloadImage('${image.url}')" class="action-btn">
                                        <i class="fas fa-download"></i>
                                    </button>
                                    <button onclick="shareImage('${image.url}')" class="action-btn">
                                        <i class="fas fa-share"></i>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                imageGrid.insertAdjacentHTML('beforeend', imageCard);
            });

        } catch (error) {
            console.error('Error loading images:', error);
            imageGrid.innerHTML = `
                <div class="error-message">
                    <p>Failed to load images. Please try again.</p>
                    <button onclick="loadUserImages()" class="retry-btn">Retry</button>
                </div>
            `;
        }
    }

    // Helper function to download image
    window.downloadImage = async (url) => {
        try {
            const response = await fetch(url);
            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = 'ai-generated-image.png';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(downloadUrl);
            document.body.removeChild(a);
        } catch (error) {
            console.error('Download failed:', error);
            alert('Failed to download image');
        }
    };

    // Helper function to share image
    window.shareImage = async (url) => {
        try {
            if (navigator.share) {
                await navigator.share({
                    title: 'AI Generated Image',
                    url: url
                });
            } else {
                await navigator.clipboard.writeText(url);
                alert('Image URL copied to clipboard!');
            }
        } catch (error) {
            console.error('Share failed:', error);
            alert('Failed to share image');
        }
    };

    // Load images when page loads
    await loadUserImages();

    // Add sort functionality
    const sortSelect = document.querySelector('.sort-select');
    if (sortSelect) {
        sortSelect.addEventListener('change', (e) => {
            const direction = e.target.value;
            const cards = [...imageGrid.querySelectorAll('.image-card')];
            
            cards.sort((a, b) => {
                const dateA = new Date(a.querySelector('.date').textContent);
                const dateB = new Date(b.querySelector('.date').textContent);
                return direction === 'newest' ? dateB - dateA : dateA - dateB;
            });

            imageGrid.innerHTML = '';
            cards.forEach(card => imageGrid.appendChild(card));
        });
    }
});

async function handleLogout() {
    try {
        const token = localStorage.getItem('token');
        console.log('Attempting logout with token:', token);

        const response = await fetch('/auth/logout', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });

        console.log('Logout response status:', response.status);

        // Always clear auth data and redirect
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        window.location.href = '/login.html';

    } catch (error) {
        console.error('Logout error:', error);
        // Clear auth data even if request fails
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        window.location.href = '/login.html';
    }
}