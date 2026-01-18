import os
import time
import requests
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ChesscomDownloader:
    BASE_URL = "https://api.chess.com/pub"

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    def __init__(self, output_dir: str = "./pgn_files"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def download_player_games(self, username: str, year_month: Optional[str] = None, max_archives: int = 3) -> Optional[str]:
        logger.info(f"Downloading games for player: {username}")
        try:
            archives_url = f"{self.BASE_URL}/player/{username}/games/archives"
            r = requests.get(archives_url, headers=self.HEADERS, timeout=30)
            r.raise_for_status()
            archives = r.json().get("archives", [])
            if not archives:
                logger.warning(f"No games found for player: {username}")
                return None
            if year_month:
                target = f"{self.BASE_URL}/player/{username}/games/{year_month}"
                if target not in archives:
                    logger.error(f"No games for {username} in {year_month}")
                    return None
                archives = [target]
            else:
                archives = archives[-max_archives:]
            filename = f"{username}_chesscom.pgn"
            filepath = os.path.join(self.output_dir, filename)
            total_games = 0
            with open(filepath, "w", encoding="utf-8") as f:
                for archive_url in archives:
                    logger.info(f"Fetching {archive_url}")
                    r = requests.get(archive_url, headers=self.HEADERS, timeout=30)
                    r.raise_for_status()
                    games = r.json().get("games", [])
                    for game in games:
                        if "pgn" in game:
                            f.write(game["pgn"])
                            f.write("\n\n")
                            total_games += 1
                    time.sleep(1)
            logger.info(f"{username}: {total_games} games → {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to download games for {username}: {e}")
            return None

    def download_top_players(self, num_players: int = 5) -> List[str]:
        logger.info("Fetching top Grandmasters...")
        try:
            url = f"{self.BASE_URL}/titled/GM"
            r = requests.get(url, headers=self.HEADERS, timeout=30)
            r.raise_for_status()
            players = r.json().get("players", [])[:num_players]
            files = []
            for player in players:
                logger.info(f"Downloading {player}...")
                path = self.download_player_games(player)
                if path:
                    files.append(path)
                time.sleep(2)
            logger.info(f"Downloaded {len(files)} GM datasets")
            return files
        except Exception as e:
            logger.error(f"Failed to download top players: {e}")
            return []


def download_chessdotcom_dataset(output_dir: str = "./pgn_files", player: Optional[str] = None) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    logger.info("=" * 60)
    logger.info("DOWNLOADING FROM CHESS.COM")
    logger.info("=" * 60)
    downloader = ChesscomDownloader(output_dir)
    files = []
    if player:
        logger.info(f"Downloading games from player: {player}")
        path = downloader.download_player_games(player)
        if path:
            files.append(path)
    else:
        logger.info("Downloading games from top Grandmasters...")
        files = downloader.download_top_players(num_players=3)
    logger.info("=" * 60)
    logger.info(f"Downloaded {len(files)} file(s)")
    logger.info(f"Location: {output_dir}")
    logger.info("=" * 60)
    return files
