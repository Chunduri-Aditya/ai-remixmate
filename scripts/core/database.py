"""
SQLite Database for AI RemixMate

This module replaces the JSON database with a proper SQLite database for:
- Song metadata storage
- Feature vectors and embeddings
- Lyrics and semantic embeddings
- Processing history and logs
- Reproducibility and caching
"""

from __future__ import annotations
import sqlite3
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
from datetime import datetime

from .features import FeatureVector


@dataclass
class SongRecord:
    """Database record for a song."""
    id: int
    name: str
    file_path: str
    file_hash: str
    bpm: float
    key: str
    camelot_number: int
    duration: float
    sample_rate: int
    bit_depth: int
    mean_mfcc: str  # JSON serialized
    chroma_centroid: str  # JSON serialized
    lyrics_text: Optional[str]
    lyrics_embedding_checksum: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class ProcessingLog:
    """Log entry for processing operations."""
    id: int
    song_id: int
    operation: str
    parameters: str  # JSON serialized
    success: bool
    error_message: Optional[str]
    processing_time: float
    timestamp: str


class RemixMateDatabase:
    """SQLite database for AI RemixMate."""
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path("models/remixmate.db")
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Songs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS songs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    bpm REAL NOT NULL,
                    key TEXT NOT NULL,
                    camelot_number INTEGER NOT NULL,
                    duration REAL NOT NULL,
                    sample_rate INTEGER NOT NULL,
                    bit_depth INTEGER NOT NULL,
                    mean_mfcc TEXT NOT NULL,
                    chroma_centroid TEXT NOT NULL,
                    lyrics_text TEXT,
                    lyrics_embedding_checksum TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Processing logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    song_id INTEGER NOT NULL,
                    operation TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    processing_time REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (song_id) REFERENCES songs (id)
                )
            """)
            
            # Remix results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS remix_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    base_song_id INTEGER NOT NULL,
                    match_song_id INTEGER NOT NULL,
                    output_path TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    metrics TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    processing_time REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (base_song_id) REFERENCES songs (id),
                    FOREIGN KEY (match_song_id) REFERENCES songs (id)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_songs_name ON songs (name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_songs_bpm ON songs (bpm)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_songs_key ON songs (key)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_songs_camelot ON songs (camelot_number)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_logs_song_id ON processing_logs (song_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_remix_results_base_song ON remix_results (base_song_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_remix_results_match_song ON remix_results (match_song_id)")
            
            conn.commit()
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""
    
    def _serialize_array(self, array: np.ndarray) -> str:
        """Serialize numpy array to JSON string."""
        return json.dumps(array.tolist())
    
    def _deserialize_array(self, json_str: str) -> np.ndarray:
        """Deserialize JSON string to numpy array."""
        return np.array(json.loads(json_str))
    
    def add_song(self, name: str, file_path: Path, features: FeatureVector, 
                lyrics_text: Optional[str] = None, lyrics_embedding_checksum: Optional[str] = None) -> int:
        """
        Add a song to the database.
        
        Args:
            name: Song name
            file_path: Path to audio file
            features: Extracted audio features
            lyrics_text: Lyrics text (optional)
            lyrics_embedding_checksum: Lyrics embedding checksum (optional)
            
        Returns:
            Song ID
        """
        file_hash = self._calculate_file_hash(file_path)
        timestamp = datetime.now().isoformat()
        
        # Detect key (simplified)
        key = "C"  # Default, would be detected from features
        camelot_number = 8  # Default for C major
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if song already exists
            cursor.execute("SELECT id FROM songs WHERE name = ?", (name,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                cursor.execute("""
                    UPDATE songs SET
                        file_path = ?,
                        file_hash = ?,
                        bpm = ?,
                        key = ?,
                        camelot_number = ?,
                        duration = ?,
                        sample_rate = ?,
                        bit_depth = ?,
                        mean_mfcc = ?,
                        chroma_centroid = ?,
                        lyrics_text = ?,
                        lyrics_embedding_checksum = ?,
                        updated_at = ?
                    WHERE name = ?
                """, (
                    str(file_path), file_hash, features.tempo, key, camelot_number,
                    0.0, 44100, 24,  # Placeholder values
                    self._serialize_array(features.mfcc),
                    self._serialize_array(features.chroma),
                    lyrics_text, lyrics_embedding_checksum, timestamp, name
                ))
                song_id = existing[0]
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO songs (
                        name, file_path, file_hash, bpm, key, camelot_number,
                        duration, sample_rate, bit_depth, mean_mfcc, chroma_centroid,
                        lyrics_text, lyrics_embedding_checksum, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    name, str(file_path), file_hash, features.tempo, key, camelot_number,
                    0.0, 44100, 24,  # Placeholder values
                    self._serialize_array(features.mfcc),
                    self._serialize_array(features.chroma),
                    lyrics_text, lyrics_embedding_checksum, timestamp, timestamp
                ))
                song_id = cursor.lastrowid
            
            conn.commit()
            return song_id
    
    def get_song(self, name: str) -> Optional[SongRecord]:
        """Get song record by name."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM songs WHERE name = ?", (name,))
            row = cursor.fetchone()
            
            if row:
                return SongRecord(
                    id=row[0], name=row[1], file_path=row[2], file_hash=row[3],
                    bpm=row[4], key=row[5], camelot_number=row[6], duration=row[7],
                    sample_rate=row[8], bit_depth=row[9], mean_mfcc=row[10],
                    chroma_centroid=row[11], lyrics_text=row[12],
                    lyrics_embedding_checksum=row[13], created_at=row[14], updated_at=row[15]
                )
            return None
    
    def get_song_features(self, name: str) -> Optional[FeatureVector]:
        """Get song features by name."""
        song = self.get_song(name)
        if song:
            return FeatureVector(
                tempo=song.bpm,
                mfcc=self._deserialize_array(song.mean_mfcc),
                chroma=self._deserialize_array(song.chroma_centroid)
            )
        return None
    
    def find_similar_songs(self, query_features: FeatureVector, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar songs using feature similarity.
        
        Args:
            query_features: Query song features
            top_k: Number of results to return
            
        Returns:
            List of (song_name, similarity_score) tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, mean_mfcc, chroma_centroid FROM songs")
            rows = cursor.fetchall()
        
        similarities = []
        query_vec = np.concatenate([query_features.chroma, query_features.mfcc])
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        
        for row in rows:
            name, mfcc_json, chroma_json = row
            try:
                mfcc = self._deserialize_array(mfcc_json)
                chroma = self._deserialize_array(chroma_json)
                candidate_vec = np.concatenate([chroma, mfcc])
                candidate_vec = candidate_vec / (np.linalg.norm(candidate_vec) + 1e-10)
                
                similarity = np.dot(query_vec, candidate_vec)
                similarities.append((name, float(similarity)))
            except Exception:
                continue
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def find_songs_by_key(self, key: str) -> List[str]:
        """Find songs by musical key."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM songs WHERE key = ?", (key,))
            return [row[0] for row in cursor.fetchall()]
    
    def find_songs_by_tempo_range(self, min_bpm: float, max_bpm: float) -> List[str]:
        """Find songs within tempo range."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM songs WHERE bpm BETWEEN ? AND ?", (min_bpm, max_bpm))
            return [row[0] for row in cursor.fetchall()]
    
    def find_songs_by_camelot_compatibility(self, camelot_number: int, max_distance: int = 2) -> List[Tuple[str, int]]:
        """Find songs compatible on Camelot wheel."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, camelot_number FROM songs")
            rows = cursor.fetchall()
        
        compatible = []
        for name, song_camelot in rows:
            # Calculate circular distance
            distance = min(abs(camelot_number - song_camelot), 
                          12 - abs(camelot_number - song_camelot))
            if distance <= max_distance:
                compatible.append((name, distance))
        
        compatible.sort(key=lambda x: x[1])  # Sort by distance
        return compatible
    
    def log_processing(self, song_id: int, operation: str, parameters: Dict, 
                      success: bool, error_message: Optional[str] = None, 
                      processing_time: float = 0.0) -> None:
        """Log a processing operation."""
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO processing_logs (
                    song_id, operation, parameters, success, error_message,
                    processing_time, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                song_id, operation, json.dumps(parameters), success,
                error_message, processing_time, timestamp
            ))
            conn.commit()
    
    def log_remix_result(self, base_song_id: int, match_song_id: int, 
                        output_path: Path, quality_score: float, 
                        metrics: Dict, parameters: Dict, processing_time: float) -> int:
        """Log a remix result."""
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO remix_results (
                    base_song_id, match_song_id, output_path, quality_score,
                    metrics, parameters, processing_time, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                base_song_id, match_song_id, str(output_path), quality_score,
                json.dumps(metrics), json.dumps(parameters), processing_time, timestamp
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_processing_history(self, song_name: str) -> List[ProcessingLog]:
        """Get processing history for a song."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT pl.* FROM processing_logs pl
                JOIN songs s ON pl.song_id = s.id
                WHERE s.name = ?
                ORDER BY pl.timestamp DESC
            """, (song_name,))
            rows = cursor.fetchall()
        
        logs = []
        for row in rows:
            logs.append(ProcessingLog(
                id=row[0], song_id=row[1], operation=row[2], parameters=row[3],
                success=bool(row[4]), error_message=row[5], processing_time=row[6],
                timestamp=row[7]
            ))
        
        return logs
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count songs
            cursor.execute("SELECT COUNT(*) FROM songs")
            song_count = cursor.fetchone()[0]
            
            # Count processing logs
            cursor.execute("SELECT COUNT(*) FROM processing_logs")
            log_count = cursor.fetchone()[0]
            
            # Count remix results
            cursor.execute("SELECT COUNT(*) FROM remix_results")
            remix_count = cursor.fetchone()[0]
            
            # Average quality score
            cursor.execute("SELECT AVG(quality_score) FROM remix_results")
            avg_quality = cursor.fetchone()[0] or 0.0
            
            # Database size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                'song_count': song_count,
                'log_count': log_count,
                'remix_count': remix_count,
                'average_quality_score': avg_quality,
                'database_size_bytes': db_size,
                'database_size_mb': db_size / (1024 * 1024)
            }
    
    def cleanup_old_logs(self, days_old: int = 30) -> int:
        """Clean up old processing logs."""
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        cutoff_iso = datetime.fromtimestamp(cutoff_date).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM processing_logs WHERE timestamp < ?", (cutoff_iso,))
            deleted_count = cursor.rowcount
            conn.commit()
        
        return deleted_count
    
    def export_to_json(self, output_path: Path) -> None:
        """Export database to JSON for backup."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Export songs
            cursor.execute("SELECT * FROM songs")
            songs = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
            
            # Export processing logs
            cursor.execute("SELECT * FROM processing_logs")
            logs = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
            
            # Export remix results
            cursor.execute("SELECT * FROM remix_results")
            remixes = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'songs': songs,
            'processing_logs': logs,
            'remix_results': remixes,
            'stats': self.get_database_stats()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Database exported to: {output_path}")
    
    def print_database_summary(self) -> None:
        """Print database summary."""
        stats = self.get_database_stats()
        
        print("\n" + "="*60)
        print("🗄️  REMIXMATE DATABASE SUMMARY")
        print("="*60)
        
        print(f"\n📊 STATISTICS")
        print(f"   Songs: {stats['song_count']}")
        print(f"   Processing logs: {stats['log_count']}")
        print(f"   Remix results: {stats['remix_count']}")
        print(f"   Average quality score: {stats['average_quality_score']:.3f}")
        print(f"   Database size: {stats['database_size_mb']:.2f} MB")
        
        print(f"\n📁 DATABASE LOCATION")
        print(f"   Path: {self.db_path}")
        print(f"   Exists: {'✅' if self.db_path.exists() else '❌'}")
        
        print("\n" + "="*60)
