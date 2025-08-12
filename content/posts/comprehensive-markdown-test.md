---
title: "Comprehensive Markdown and Code Highlighting Test"
date: 2024-12-08T14:30:00Z
draft: false
tags: ["markdown", "syntax-highlighting", "monokai", "testing", "programming"]
categories: ["testing", "documentation"]
description: "A comprehensive test of all Markdown features and syntax highlighting capabilities with the Monokai theme"
showToc: true
TocOpen: false
hidemeta: false
comments: false
disableHLJS: false
disableShare: false
searchHidden: false
cover:
    image: ""
    alt: ""
    caption: ""
    relative: false
    hidden: true
---

This post serves as a comprehensive test of all Markdown features and syntax highlighting capabilities using the Monokai color scheme. It demonstrates various programming languages, formatting options, and technical content elements.

## Text Formatting

**Bold text** and *italic text* and ***bold italic text***. You can also use __bold__ and _italic_ syntax.

~~Strikethrough text~~ and `inline code` with backticks.

Here's a line with superscript: E = mc² and subscript: H₂O.

## Headers and Structure

### Level 3 Header
#### Level 4 Header
##### Level 5 Header
###### Level 6 Header

## Programming Languages Testing

### Python - Data Science Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def analyze_data(data_file):
    """
    Comprehensive data analysis function with error handling.
    
    Args:
        data_file (str): Path to the CSV data file
        
    Returns:
        dict: Analysis results including model metrics
    """
    try:
        # Load and preprocess data
        df = pd.read_csv(data_file)
        df = df.dropna()  # Remove missing values
        
        # Feature engineering
        X = df[['feature1', 'feature2', 'feature3']]
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'r2': r2,
            'model': model,
            'predictions': y_pred
        }
        
    except Exception as e:
        print(f"Error in data analysis: {e}")
        return None

# Usage example
if __name__ == "__main__":
    results = analyze_data("sample_data.csv")
    if results:
        print(f"Model R² Score: {results['r2']:.4f}")
        print(f"Mean Squared Error: {results['mse']:.4f}")
```

### JavaScript - Modern Web Development

```javascript
// Modern JavaScript with ES6+ features
class APIClient {
    constructor(baseURL, apiKey) {
        this.baseURL = baseURL;
        this.apiKey = apiKey;
        this.headers = {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
        };
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: this.headers,
            ...options
        };

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            }
            
            return await response.text();
        } catch (error) {
            console.error(`API request failed: ${error.message}`);
            throw error;
        }
    }

    // CRUD operations
    async get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    }

    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    async put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }
}

// Usage with async/await and error handling
const client = new APIClient('https://api.example.com', 'your-api-key');

async function fetchUserProfile(userId) {
    try {
        const user = await client.get(`/users/${userId}`);
        const posts = await client.get(`/users/${userId}/posts`);
        
        return {
            ...user,
            posts: posts.data || []
        };
    } catch (error) {
        console.error('Failed to fetch user profile:', error);
        return null;
    }
}

// Modern array methods and destructuring
const processUsers = (users) => {
    return users
        .filter(user => user.active)
        .map(({ id, name, email, ...rest }) => ({
            id,
            displayName: name,
            contact: email,
            metadata: rest
        }))
        .sort((a, b) => a.displayName.localeCompare(b.displayName));
};
```

### Go - Concurrent Web Server

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "os"
    "os/signal"
    "sync"
    "syscall"
    "time"
)

// User represents a user in our system
type User struct {
    ID       int    `json:"id"`
    Name     string `json:"name"`
    Email    string `json:"email"`
    Created  time.Time `json:"created"`
}

// UserStore provides thread-safe user storage
type UserStore struct {
    mu    sync.RWMutex
    users map[int]*User
    nextID int
}

func NewUserStore() *UserStore {
    return &UserStore{
        users: make(map[int]*User),
        nextID: 1,
    }
}

func (s *UserStore) Create(name, email string) *User {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    user := &User{
        ID:      s.nextID,
        Name:    name,
        Email:   email,
        Created: time.Now(),
    }
    
    s.users[s.nextID] = user
    s.nextID++
    
    return user
}

func (s *UserStore) Get(id int) (*User, bool) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    user, exists := s.users[id]
    return user, exists
}

func (s *UserStore) List() []*User {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    users := make([]*User, 0, len(s.users))
    for _, user := range s.users {
        users = append(users, user)
    }
    
    return users
}

// Server handles HTTP requests
type Server struct {
    store  *UserStore
    server *http.Server
}

func NewServer(addr string) *Server {
    store := NewUserStore()
    
    mux := http.NewServeMux()
    server := &Server{
        store: store,
        server: &http.Server{
            Addr:    addr,
            Handler: mux,
        },
    }
    
    // Register routes
    mux.HandleFunc("/users", server.handleUsers)
    mux.HandleFunc("/users/", server.handleUser)
    mux.HandleFunc("/health", server.handleHealth)
    
    return server
}

func (s *Server) handleUsers(w http.ResponseWriter, r *http.Request) {
    switch r.Method {
    case http.MethodGet:
        users := s.store.List()
        s.writeJSON(w, users)
    case http.MethodPost:
        var req struct {
            Name  string `json:"name"`
            Email string `json:"email"`
        }
        
        if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
            http.Error(w, "Invalid JSON", http.StatusBadRequest)
            return
        }
        
        user := s.store.Create(req.Name, req.Email)
        w.WriteHeader(http.StatusCreated)
        s.writeJSON(w, user)
    default:
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
    }
}

func (s *Server) handleUser(w http.ResponseWriter, r *http.Request) {
    // Implementation for individual user operations
    w.WriteHeader(http.StatusNotImplemented)
    fmt.Fprintf(w, "Individual user operations not implemented yet")
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
    response := map[string]interface{}{
        "status":    "healthy",
        "timestamp": time.Now(),
        "users":     len(s.store.List()),
    }
    s.writeJSON(w, response)
}

func (s *Server) writeJSON(w http.ResponseWriter, data interface{}) {
    w.Header().Set("Content-Type", "application/json")
    if err := json.NewEncoder(w).Encode(data); err != nil {
        log.Printf("Error encoding JSON: %v", err)
        http.Error(w, "Internal server error", http.StatusInternalServerError)
    }
}

func (s *Server) Start() error {
    log.Printf("Starting server on %s", s.server.Addr)
    return s.server.ListenAndServe()
}

func (s *Server) Shutdown(ctx context.Context) error {
    log.Println("Shutting down server...")
    return s.server.Shutdown(ctx)
}

func main() {
    server := NewServer(":8080")
    
    // Start server in goroutine
    go func() {
        if err := server.Start(); err != nil && err != http.ErrServerClosed {
            log.Fatalf("Server failed to start: %v", err)
        }
    }()
    
    // Wait for interrupt signal
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit
    
    // Graceful shutdown
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    if err := server.Shutdown(ctx); err != nil {
        log.Fatalf("Server forced to shutdown: %v", err)
    }
    
    log.Println("Server exited")
}
```

### Rust - Systems Programming

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tokio::time::sleep;

// Generic cache implementation with TTL
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    value: T,
    expires_at: Instant,
}

impl<T> CacheEntry<T> {
    fn new(value: T, ttl: Duration) -> Self {
        Self {
            value,
            expires_at: Instant::now() + ttl,
        }
    }
    
    fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

pub struct Cache<K, V> 
where 
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    store: Arc<Mutex<HashMap<K, CacheEntry<V>>>>,
    default_ttl: Duration,
}

impl<K, V> Cache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    pub fn new(default_ttl: Duration) -> Self {
        Self {
            store: Arc::new(Mutex::new(HashMap::new())),
            default_ttl,
        }
    }
    
    pub fn insert(&self, key: K, value: V) -> Result<(), String> {
        self.insert_with_ttl(key, value, self.default_ttl)
    }
    
    pub fn insert_with_ttl(&self, key: K, value: V, ttl: Duration) -> Result<(), String> {
        let mut store = self.store.lock().map_err(|e| format!("Lock error: {}", e))?;
        let entry = CacheEntry::new(value, ttl);
        store.insert(key, entry);
        Ok(())
    }
    
    pub fn get(&self, key: &K) -> Result<Option<V>, String> {
        let mut store = self.store.lock().map_err(|e| format!("Lock error: {}", e))?;
        
        match store.get(key) {
            Some(entry) if !entry.is_expired() => Ok(Some(entry.value.clone())),
            Some(_) => {
                // Remove expired entry
                store.remove(key);
                Ok(None)
            }
            None => Ok(None),
        }
    }
    
    pub fn remove(&self, key: &K) -> Result<Option<V>, String> {
        let mut store = self.store.lock().map_err(|e| format!("Lock error: {}", e))?;
        Ok(store.remove(key).map(|entry| entry.value))
    }
    
    pub fn cleanup_expired(&self) -> Result<usize, String> {
        let mut store = self.store.lock().map_err(|e| format!("Lock error: {}", e))?;
        let initial_size = store.len();
        
        store.retain(|_, entry| !entry.is_expired());
        
        Ok(initial_size - store.len())
    }
    
    pub fn size(&self) -> Result<usize, String> {
        let store = self.store.lock().map_err(|e| format!("Lock error: {}", e))?;
        Ok(store.len())
    }
}

// Async function example
async fn fetch_data_with_cache(
    cache: &Cache<String, String>,
    key: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    // Try to get from cache first
    if let Some(cached_value) = cache.get(&key.to_string())? {
        println!("Cache hit for key: {}", key);
        return Ok(cached_value);
    }
    
    println!("Cache miss for key: {}, fetching...", key);
    
    // Simulate async data fetching
    sleep(Duration::from_millis(100)).await;
    let data = format!("Data for {}", key);
    
    // Store in cache
    cache.insert(key.to_string(), data.clone())?;
    
    Ok(data)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = Cache::new(Duration::from_secs(5));
    
    // Test cache operations
    println!("Testing cache operations...");
    
    // Insert some data
    cache.insert("user:1".to_string(), "John Doe".to_string())?;
    cache.insert("user:2".to_string(), "Jane Smith".to_string())?;
    
    // Retrieve data
    if let Some(user) = cache.get(&"user:1".to_string())? {
        println!("Found user: {}", user);
    }
    
    // Test async function with cache
    let result1 = fetch_data_with_cache(&cache, "async_key").await?;
    let result2 = fetch_data_with_cache(&cache, "async_key").await?; // Should hit cache
    
    println!("First fetch: {}", result1);
    println!("Second fetch: {}", result2);
    
    // Spawn cleanup task
    let cache_clone = cache.clone();
    let cleanup_handle = tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(10)).await;
            match cache_clone.cleanup_expired() {
                Ok(removed) => {
                    if removed > 0 {
                        println!("Cleaned up {} expired entries", removed);
                    }
                }
                Err(e) => eprintln!("Cleanup error: {}", e),
            }
        }
    });
    
    // Let the program run for a bit
    sleep(Duration::from_secs(2)).await;
    
    println!("Cache size: {}", cache.size()?);
    
    // Cancel cleanup task
    cleanup_handle.abort();
    
    Ok(())
}
```

### TypeScript - Advanced Type System

```typescript
// Advanced TypeScript with generics and utility types
interface BaseEntity {
    id: string;
    createdAt: Date;
    updatedAt: Date;
}

interface User extends BaseEntity {
    name: string;
    email: string;
    role: 'admin' | 'user' | 'moderator';
    preferences: UserPreferences;
}

interface UserPreferences {
    theme: 'light' | 'dark' | 'auto';
    notifications: {
        email: boolean;
        push: boolean;
        sms: boolean;
    };
    privacy: {
        profileVisible: boolean;
        activityVisible: boolean;
    };
}

// Generic repository pattern
abstract class Repository<T extends BaseEntity> {
    protected abstract tableName: string;
    
    abstract findById(id: string): Promise<T | null>;
    abstract findAll(filters?: Partial<T>): Promise<T[]>;
    abstract create(data: Omit<T, keyof BaseEntity>): Promise<T>;
    abstract update(id: string, data: Partial<Omit<T, keyof BaseEntity>>): Promise<T>;
    abstract delete(id: string): Promise<boolean>;
    
    // Common validation method
    protected validateEntity(entity: Partial<T>): string[] {
        const errors: string[] = [];
        
        if (!entity.id && this.isUpdate(entity)) {
            errors.push('ID is required for updates');
        }
        
        return errors;
    }
    
    private isUpdate(entity: Partial<T>): entity is T {
        return 'id' in entity && entity.id !== undefined;
    }
}

// Concrete implementation
class UserRepository extends Repository<User> {
    protected tableName = 'users';
    
    async findById(id: string): Promise<User | null> {
        // Simulate database query
        const mockUser: User = {
            id,
            name: 'John Doe',
            email: 'john@example.com',
            role: 'user',
            preferences: {
                theme: 'dark',
                notifications: {
                    email: true,
                    push: false,
                    sms: false,
                },
                privacy: {
                    profileVisible: true,
                    activityVisible: false,
                },
            },
            createdAt: new Date(),
            updatedAt: new Date(),
        };
        
        return mockUser;
    }
    
    async findAll(filters?: Partial<User>): Promise<User[]> {
        // Implementation would query database with filters
        return [];
    }
    
    async create(data: Omit<User, keyof BaseEntity>): Promise<User> {
        const now = new Date();
        const user: User = {
            ...data,
            id: crypto.randomUUID(),
            createdAt: now,
            updatedAt: now,
        };
        
        // Save to database
        return user;
    }
    
    async update(id: string, data: Partial<Omit<User, keyof BaseEntity>>): Promise<User> {
        const existingUser = await this.findById(id);
        if (!existingUser) {
            throw new Error(`User with id ${id} not found`);
        }
        
        const updatedUser: User = {
            ...existingUser,
            ...data,
            updatedAt: new Date(),
        };
        
        // Save to database
        return updatedUser;
    }
    
    async delete(id: string): Promise<boolean> {
        // Implementation would delete from database
        return true;
    }
    
    // User-specific methods
    async findByEmail(email: string): Promise<User | null> {
        // Implementation would query by email
        return null;
    }
    
    async findByRole(role: User['role']): Promise<User[]> {
        // Implementation would filter by role
        return [];
    }
}

// Service layer with dependency injection
class UserService {
    constructor(private userRepository: UserRepository) {}
    
    async createUser(userData: {
        name: string;
        email: string;
        role?: User['role'];
    }): Promise<User> {
        // Validation
        if (!userData.name || !userData.email) {
            throw new Error('Name and email are required');
        }
        
        // Check if user already exists
        const existingUser = await this.userRepository.findByEmail(userData.email);
        if (existingUser) {
            throw new Error('User with this email already exists');
        }
        
        // Create user with default preferences
        const newUser = await this.userRepository.create({
            name: userData.name,
            email: userData.email,
            role: userData.role || 'user',
            preferences: {
                theme: 'auto',
                notifications: {
                    email: true,
                    push: true,
                    sms: false,
                },
                privacy: {
                    profileVisible: true,
                    activityVisible: true,
                },
            },
        });
        
        return newUser;
    }
    
    async updateUserPreferences(
        userId: string,
        preferences: Partial<UserPreferences>
    ): Promise<User> {
        const user = await this.userRepository.findById(userId);
        if (!user) {
            throw new Error('User not found');
        }
        
        const updatedPreferences: UserPreferences = {
            ...user.preferences,
            ...preferences,
            notifications: {
                ...user.preferences.notifications,
                ...preferences.notifications,
            },
            privacy: {
                ...user.preferences.privacy,
                ...preferences.privacy,
            },
        };
        
        return this.userRepository.update(userId, {
            preferences: updatedPreferences,
        });
    }
}

// Usage example with error handling
async function main() {
    const userRepository = new UserRepository();
    const userService = new UserService(userRepository);
    
    try {
        // Create a new user
        const newUser = await userService.createUser({
            name: 'Alice Johnson',
            email: 'alice@example.com',
            role: 'moderator',
        });
        
        console.log('Created user:', newUser);
        
        // Update user preferences
        const updatedUser = await userService.updateUserPreferences(newUser.id, {
            theme: 'dark',
            notifications: {
                push: false,
            },
        });
        
        console.log('Updated user preferences:', updatedUser.preferences);
        
    } catch (error) {
        console.error('Error:', error instanceof Error ? error.message : error);
    }
}

// Type utilities and advanced patterns
type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

type UserWithRequiredEmail = RequiredFields<Partial<User>, 'email'>;

// Conditional types
type ApiResponse<T> = T extends string
    ? { message: T }
    : T extends object
    ? { data: T }
    : { value: T };

// Template literal types
type EventName<T extends string> = `on${Capitalize<T>}`;
type UserEvents = EventName<'create' | 'update' | 'delete'>;

main().catch(console.error);
```

## Lists and Structure

### Ordered Lists

1. First item with **bold text**
2. Second item with *italic text*
3. Third item with `inline code`
   1. Nested item A
   2. Nested item B
      1. Double nested item
4. Fourth item with [a link](https://example.com)

### Unordered Lists

- Bullet point one
- Bullet point two
  - Nested bullet A
  - Nested bullet B
    - Double nested bullet
- Bullet point three

### Task Lists

- [x] Completed task
- [x] Another completed task
- [ ] Incomplete task
- [ ] Another incomplete task
  - [x] Nested completed subtask
  - [ ] Nested incomplete subtask

## Tables

### Programming Language Comparison

| Language   | Type System | Memory Management | Concurrency Model | Performance |
|------------|-------------|-------------------|-------------------|-------------|
| Python     | Dynamic     | Garbage Collected | GIL + Threading   | Moderate    |
| JavaScript | Dynamic     | Garbage Collected | Event Loop        | Good        |
| Go         | Static      | Garbage Collected | Goroutines        | Excellent   |
| Rust       | Static      | Manual/RAII       | Async/Await       | Excellent   |
| TypeScript | Static      | Garbage Collected | Event Loop        | Good        |

### Feature Matrix

| Feature              | Hugo | Jekyll | Gatsby | Next.js |
|---------------------|------|--------|--------|---------|
| Build Speed         | ⚡⚡⚡  | ⚡     | ⚡⚡    | ⚡⚡     |
| Learning Curve      | Easy | Easy   | Hard   | Medium  |
| Plugin Ecosystem    | Good | Great  | Great  | Great   |
| Static Generation   | ✅   | ✅     | ✅     | ✅      |
| Server-Side Rendering| ❌   | ❌     | ❌     | ✅      |

## Blockquotes and Callouts

> **Note**: This is a standard blockquote that can contain multiple paragraphs and other elements.
>
> It can span multiple lines and include `inline code` and **formatting**.

> **Warning**: This represents important information that users should pay attention to.

> **Tip**: Pro tip for developers - always test your code with edge cases!

## Links and References

Here are various types of links:

- [External link](https://www.example.com)
- [Link with title](https://www.example.com "Example Website")
- [Reference-style link][ref1]
- [Another reference link][ref2]
- Email link: <email@example.com>
- Automatic link: <https://www.example.com>

[ref1]: https://www.example.com "Reference 1"
[ref2]: https://www.github.com "GitHub"

## Images and Media

![Alt text for image](https://via.placeholder.com/600x300/272822/F8F8F2?text=Monokai+Theme+Test)

*Caption: This is a sample image demonstrating the Monokai color scheme*

## Mathematical Expressions

If MathJax is enabled, these should render properly:

Inline math: $E = mc^2$ and $\sum_{i=1}^{n} x_i$

Block math:
$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$

$$
\begin{align}
\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} &= \frac{4\pi}{c}\vec{\mathbf{j}} \\
\nabla \cdot \vec{\mathbf{E}} &= 4 \pi \rho \\
\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t} &= \vec{\mathbf{0}} \\
\nabla \cdot \vec{\mathbf{B}} &= 0
\end{align}
$$

## Code Blocks with Different Languages

### SQL Database Query

```sql
-- Complex SQL query with joins and aggregations
SELECT 
    u.id,
    u.name,
    u.email,
    COUNT(p.id) as post_count,
    AVG(p.rating) as avg_rating,
    MAX(p.created_at) as last_post_date
FROM users u
LEFT JOIN posts p ON u.id = p.author_id
WHERE u.active = true
    AND u.created_at >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
GROUP BY u.id, u.name, u.email
HAVING COUNT(p.id) > 5
ORDER BY avg_rating DESC, post_count DESC
LIMIT 50;

-- Create index for performance
CREATE INDEX idx_posts_author_created ON posts(author_id, created_at);

-- Update user statistics
UPDATE users 
SET last_login = NOW(),
    login_count = login_count + 1
WHERE id = ?;
```

### Bash/Shell Scripting

```bash
#!/bin/bash

# Hugo blog deployment script
set -euo pipefail

# Configuration
HUGO_VERSION="0.120.0"
SITE_DIR="hugo-blog"
PUBLIC_DIR="public"
DEPLOY_BRANCH="gh-pages"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Hugo is installed
check_hugo() {
    if ! command -v hugo &> /dev/null; then
        log_error "Hugo is not installed. Installing Hugo v${HUGO_VERSION}..."
        
        # Install Hugo based on OS
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install hugo
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            wget -O hugo.deb "https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-amd64.deb"
            sudo dpkg -i hugo.deb
            rm hugo.deb
        else
            log_error "Unsupported operating system"
            exit 1
        fi
    fi
    
    log_info "Hugo version: $(hugo version)"
}

# Build the site
build_site() {
    log_info "Building Hugo site..."
    
    cd "$SITE_DIR"
    
    # Clean previous build
    if [ -d "$PUBLIC_DIR" ]; then
        rm -rf "$PUBLIC_DIR"
    fi
    
    # Build with Hugo
    hugo --minify --gc
    
    if [ $? -eq 0 ]; then
        log_info "Site built successfully"
    else
        log_error "Site build failed"
        exit 1
    fi
    
    cd ..
}

# Deploy to GitHub Pages
deploy_site() {
    log_info "Deploying to GitHub Pages..."
    
    # Check if we're in a git repository
    if [ ! -d ".git" ]; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Create temporary directory for deployment
    TEMP_DIR=$(mktemp -d)
    
    # Copy built site to temp directory
    cp -r "${SITE_DIR}/${PUBLIC_DIR}/"* "$TEMP_DIR/"
    
    # Switch to deployment branch
    git checkout "$DEPLOY_BRANCH" 2>/dev/null || git checkout -b "$DEPLOY_BRANCH"
    
    # Clear current content (except .git)
    find . -maxdepth 1 ! -name '.git' ! -name '.' -exec rm -rf {} +
    
    # Copy new content
    cp -r "$TEMP_DIR/"* .
    
    # Add CNAME file if needed
    if [ ! -z "${CUSTOM_DOMAIN:-}" ]; then
        echo "$CUSTOM_DOMAIN" > CNAME
    fi
    
    # Commit and push
    git add .
    git commit -m "Deploy site - $(date)"
    git push origin "$DEPLOY_BRANCH"
    
    # Switch back to main branch
    git checkout main
    
    # Cleanup
    rm -rf "$TEMP_DIR"
    
    log_info "Deployment completed successfully"
}

# Main execution
main() {
    log_info "Starting Hugo blog deployment..."
    
    check_hugo
    build_site
    deploy_site
    
    log_info "All done! 🎉"
}

# Run main function
main "$@"
```

### YAML Configuration

```yaml
# Hugo configuration file
baseURL: 'https://yourblog.github.io'
languageCode: 'en-us'
title: 'Technical Blog'
theme: 'PaperMod'

# Enable syntax highlighting
markup:
  highlight:
    style: 'monokai'
    lineNos: true
    lineNumbersInTable: false
    noClasses: false
    anchorLineNos: false
    codeFences: true
    guessSyntax: false
    hl_Lines: ''
    hl_inline: false
    tabWidth: 4

# Site parameters
params:
  author: 'Your Name'
  description: 'A technical blog about programming and software development'
  keywords: ['programming', 'software', 'development', 'tutorial']
  
  # Theme settings
  ShowReadingTime: true
  ShowShareButtons: true
  ShowPostNavLinks: true
  ShowBreadCrumbs: true
  ShowCodeCopyButtons: true
  ShowWordCount: true
  ShowRssButtonInSectionTermList: true
  UseHugoToc: true
  disableSpecial1stPost: false
  disableScrollToTop: false
  comments: false
  hidemeta: false
  hideSummary: false
  showtoc: false
  tocopen: false

  # Assets configuration
  assets:
    disableHLJS: false
    disableFingerprinting: false
    favicon: '/favicon.ico'
    favicon16x16: '/favicon-16x16.png'
    favicon32x32: '/favicon-32x32.png'
    apple_touch_icon: '/apple-touch-icon.png'
    safari_pinned_tab: '/safari-pinned-tab.svg'

  # Social media
  social:
    github: 'yourusername'
    twitter: 'yourusername'
    linkedin: 'yourusername'

# Menu configuration
menu:
  main:
    - identifier: 'home'
      name: 'Home'
      url: '/'
      weight: 10
    - identifier: 'posts'
      name: 'Posts'
      url: '/posts/'
      weight: 20
    - identifier: 'categories'
      name: 'Categories'
      url: '/categories/'
      weight: 30
    - identifier: 'tags'
      name: 'Tags'
      url: '/tags/'
      weight: 40
    - identifier: 'about'
      name: 'About'
      url: '/about/'
      weight: 50

# Taxonomies
taxonomies:
  category: 'categories'
  tag: 'tags'
  series: 'series'

# Privacy settings
privacy:
  disqus:
    disable: true
  googleAnalytics:
    disable: false
    respectDoNotTrack: true
    anonymizeIP: true
    useSessionStorage: true

# Security settings
security:
  enableInlineShortcodes: false
  exec:
    allow: ['^dart-sass-embedded$', '^go$', '^npx$', '^postcss$']
    osEnv: ['(?i)^(PATH|PATHEXT|APPDATA|TMP|TEMP|TERM)$']

# Output formats
outputs:
  home:
    - HTML
    - RSS
    - JSON
  page:
    - HTML
  section:
    - HTML
    - RSS

# Related content
related:
  includeNewer: true
  indices:
    - name: 'keywords'
      weight: 100
    - name: 'tags'
      weight: 80
    - name: 'categories'
      weight: 60
    - name: 'date'
      weight: 10
  threshold: 80
  toLower: false
```

### JSON Data Structure

```json
{
  "blog": {
    "metadata": {
      "title": "Technical Blog",
      "description": "A comprehensive technical blog with programming tutorials",
      "version": "1.0.0",
      "lastUpdated": "2025-12-08T14:30:00Z"
    },
    "configuration": {
      "theme": {
        "name": "PaperMod",
        "version": "7.0.0",
        "customizations": {
          "syntaxHighlighting": {
            "theme": "monokai",
            "lineNumbers": true,
            "copyButton": true
          },
          "colors": {
            "primary": "#F92672",
            "secondary": "#A6E22E",
            "background": "#272822",
            "text": "#F8F8F2"
          }
        }
      },
      "features": {
        "search": true,
        "comments": false,
        "analytics": true,
        "socialSharing": true,
        "tableOfContents": true,
        "readingTime": true,
        "wordCount": true
      }
    },
    "content": {
      "categories": [
        {
          "name": "Programming",
          "slug": "programming",
          "description": "Programming tutorials and best practices",
          "postCount": 15
        },
        {
          "name": "Web Development",
          "slug": "web-development", 
          "description": "Frontend and backend web development",
          "postCount": 12
        },
        {
          "name": "DevOps",
          "slug": "devops",
          "description": "Deployment, CI/CD, and infrastructure",
          "postCount": 8
        }
      ],
      "tags": [
        "javascript", "python", "go", "rust", "typescript",
        "react", "vue", "angular", "node.js", "docker",
        "kubernetes", "aws", "git", "testing", "performance"
      ],
      "recentPosts": [
        {
          "title": "Advanced TypeScript Patterns",
          "slug": "advanced-typescript-patterns",
          "publishDate": "2025-12-08",
          "readingTime": "12 min",
          "tags": ["typescript", "programming", "patterns"],
          "excerpt": "Exploring advanced TypeScript patterns for better code organization"
        },
        {
          "title": "Building Concurrent Applications in Go",
          "slug": "go-concurrency-patterns",
          "publishDate": "2025-12-07",
          "readingTime": "15 min",
          "tags": ["go", "concurrency", "programming"],
          "excerpt": "Learn how to build efficient concurrent applications using Go"
        }
      ]
    },
    "analytics": {
      "pageViews": {
        "total": 45230,
        "thisMonth": 3420,
        "thisWeek": 890
      },
      "popularPosts": [
        {
          "title": "Getting Started with Rust",
          "views": 5420,
          "engagement": 0.78
        },
        {
          "title": "Modern JavaScript Best Practices",
          "views": 4890,
          "engagement": 0.82
        }
      ],
      "traffic": {
        "sources": {
          "organic": 0.65,
          "direct": 0.20,
          "social": 0.10,
          "referral": 0.05
        },
        "devices": {
          "desktop": 0.60,
          "mobile": 0.35,
          "tablet": 0.05
        }
      }
    }
  }
}
```

## Horizontal Rules

---

Above and below this text are horizontal rules created with three dashes.

***

This one is created with three asterisks.

___

And this one with three underscores.

## Escape Characters and Special Symbols

Here are some special characters that need escaping in Markdown:

- Backslash: \\
- Backtick: \`
- Asterisk: \*
- Underscore: \_
- Curly braces: \{ \}
- Square brackets: \[ \]
- Parentheses: \( \)
- Hash: \#
- Plus: \+
- Minus: \-
- Dot: \.
- Exclamation: \!

## Conclusion

This comprehensive test post demonstrates:

✅ **Text Formatting**: Bold, italic, strikethrough, inline code
✅ **Headers**: All levels from H1 to H6
✅ **Code Blocks**: Multiple programming languages with Monokai highlighting
✅ **Lists**: Ordered, unordered, and task lists with nesting
✅ **Tables**: Complex tables with alignment and formatting
✅ **Links**: Various link types and references
✅ **Blockquotes**: Standard and styled callouts
✅ **Mathematical Expressions**: Inline and block math (if MathJax enabled)
✅ **Special Characters**: Proper escaping and symbols
✅ **Horizontal Rules**: Different styles of dividers

The Monokai syntax highlighting should be clearly visible in all code blocks, providing excellent readability for technical content. This post serves as a comprehensive test of the Hugo blog's Markdown processing capabilities and theme integration.