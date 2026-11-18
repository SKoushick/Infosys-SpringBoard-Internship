# TEDMind Database Schema Specification

This document details the relational schema implemented in Django for the TEDMind platform.

---

## 1. Entity Relationship Diagram (ERD)

```
   ┌──────────────────┐               ┌──────────────────┐
   │    auth.User     │1             1│   UserProfile    │
   ├──────────────────┼───────────────┼──────────────────┤
   │ id (PK)          │               │ user_id (FK, UQ) │
   │ username         │               │ preferred_topics │
   │ email            │               │ bio              │
   │ password         │               │ onboarding_done  │
   │ date_joined      │               │ recs_enabled     │
   └────────┬─────────┘               └──────────────────┘
            │
            │ 1:N
            ├─────────────────────────┐
            │                         │
            ▼                         ▼
   ┌──────────────────┐      ┌──────────────────┐
   │    SavedTalk     │      │    Collection    │
   ├──────────────────┤      ├──────────────────┤
   │ id (PK)          │      │ id (PK)          │
   │ user_id (FK, opt)│      │ user_id (FK)     │
   │ session_key (opt)│      │ title            │
   │ talk_id (INT)    │      │ description      │
   │ title            │      │ is_public        │
   │ speaker          │      │ created_at       │
   │ tags             │      └────────┬─────────┘
   │ created_at       │               │
   └──────────────────┘               │ 1:N
                                      ▼
                             ┌──────────────────┐
                             │  CollectionItem  │
                             ├──────────────────┤
                             │ id (PK)          │
                             │ collection_id(FK)│
                             │ talk_id (INT)    │
                             │ notes            │
                             │ added_at         │
                             └──────────────────┘
```

---

## 2. Table Definitions

### 2.1 `accounts_userprofile`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | BigAutoField | Primary Key | Auto-incrementing primary key |
| `user_id` | Integer | OneToOneField(User), Unique | Links to Django auth user |
| `bio` | Text | Max 500 chars, Blank | User bio and interests |
| `preferred_topics` | JSONField | Default `[]` | List of curated topic slugs |
| `onboarding_completed` | Boolean | Default `False` | Has completed onboarding |
| `recommendations_enabled`| Boolean | Default `True` | Algorithmic personalization toggle |
| `created_at` | DateTime | Auto now add | Timestamp |
| `updated_at` | DateTime | Auto now | Timestamp |

### 2.2 `talks_savedtalk`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | BigAutoField | Primary Key | Primary Key |
| `user_id` | Integer | ForeignKey(User), Nullable | Links to registered user |
| `session_key` | CharField(64) | Nullable, Indexed | Links to anonymous guest session |
| `talk_id` | Integer | Indexed | ID of the bookmarked TED talk |
| `title` | CharField(255) | Required | Talk presentation title |
| `speaker` | CharField(255) | Required | Main speaker name |
| `tags` | CharField(255) | Blank | Comma-separated preview tags |
| `is_liked` | Boolean | Default `True` | Positive interaction flag |
| `created_at` | DateTime | Auto now add | Creation timestamp |

### 2.3 `talks_collection` & `talks_collectionitem`
- **`talks_collection`**: User-curated playlists. Unique together on `(user_id, title)`.
- **`talks_collectionitem`**: Talk items within collections. Unique together on `(collection_id, talk_id)`.

### 2.4 `talks_viewinghistory` & `talks_userfeedback`
- **`talks_viewinghistory`**: Tracks user presentation visits with timestamps for dynamic candidate scoring.
- **`talks_userfeedback`**: Captures explicit user feedback (`like`, `dislike`, `not_interested`) to exclude disliked items.
- **`talks_recommendationevent`**: Audits query parameters, chosen model, latency, and recommended talk IDs.
