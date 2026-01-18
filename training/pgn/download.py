import os
import time
import requests
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ChesscomDownloader:
    """
    Download games từ Chess.com API
    """
    
    BASE_URL = "https://api.chess.com/pub"
    
    def __init__(self, output_dir: str = "./pgn_files"):
        """
        Args:
            output_dir: Thư mục lưu PGN files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def download_player_games(self, username: str, year_month: Optional[str] = None) -> str:
        """
        Download games của một player từ Chess.com
        
        Args:
            username: Chess.com username
            year_month: Format 'YYYY/MM' (ví dụ: '2024/01')
                       Nếu None, download tất cả
                       
        Returns:
            Filepath
        """
        logger.info(f"Downloading games for player: {username}")
        
        try:
            # Lấy danh sách archives
            url = f"{self.BASE_URL}/player/{username}/games/archives"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            archives = response.json().get('archives', [])
            
            if not archives:
                logger.warning(f"No games found for player: {username}")
                return None
            
            # Nếu có year_month, chỉ lấy cái đó
            if year_month:
                target_url = f"{self.BASE_URL}/player/{username}/games/{year_month}"
                if target_url not in archives:
                    logger.error(f"No games for {username} in {year_month}")
                    return None
                archives = [target_url]
            else:
                # Lấy 3 tháng gần nhất
                archives = archives[-3:]
            
            # Download từ mỗi archive
            filename = f"{username}_chessdotcom_games.pgn"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                for archive_url in archives:
                    logger.info(f"Downloading from {archive_url}...")
                    
                    response = requests.get(archive_url, timeout=30)
                    response.raise_for_status()
                    
                    pgn_text = response.json().get('games', [])
                    
                    for game in pgn_text:
                        # Chuyển game dict thành PGN format
                        pgn = self._convert_game_to_pgn(game)
                        f.write(pgn + '\n\n')
                    
                    # Rate limiting
                    time.sleep(1)
            
            file_size = os.path.getsize(filepath)
            logger.info(f"✅ Downloaded: {filepath} ({file_size} bytes)")
            
            return filepath
        
        except Exception as e:
            logger.error(f"Failed to download games for {username}: {e}")
            return None
    
    def _convert_game_to_pgn(self, game: dict) -> str:
        """
        Chuyển game dict từ Chess.com API sang PGN format
        """
        # Chess.com API trả về trực tiếp PGN string trong 'pgn' field
        # Không cần convert, chỉ cần lấy ra
        pgn = game.get('pgn', '')
        
        return pgn if pgn else ""
    
    def download_top_players(self, num_players: int = 5) -> List[str]:
        """
        Download games từ top players hiện tại
        
        Args:
            num_players: Số players để download
            
        Returns:
            List of filepaths
        """
        logger.info("Fetching top players from Chess.com...")
        
        try:
            url = f"{self.BASE_URL}/titled/GM"  # Grandmasters
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            players = response.json().get('players', [])[:num_players]
            
            files = []
            for player in players:
                logger.info(f"Downloading games for {player}...")
                
                filepath = self.download_player_games(player)
                
                if filepath:
                    files.append(filepath)
                
                # Rate limiting
                time.sleep(2)
            
            logger.info(f"Downloaded {len(files)} files")
            return files
        
        except Exception as e:
            logger.error(f"Failed to download top players: {e}")
            return []


def download_chessdotcom_dataset(output_dir: str = "./pgn_files",
                                player: Optional[str] = None) -> List[str]:
    """
    Download dataset từ Chess.com
    
    Args:
        output_dir: Thư mục lưu files
        player: Specific player. Nếu None, lấy top GMs
        
    Returns:
        List of filepaths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("DOWNLOADING FROM CHESS.COM")
    logger.info("=" * 60)
    
    downloader = ChesscomDownloader(output_dir)
    files = []
    
    if player:
        logger.info(f"\nDownloading games from player: {player}")
        filepath = downloader.download_player_games(player)
        if filepath:
            files.append(filepath)
    else:
        logger.info("\nDownloading games from top Grandmasters...")
        files = downloader.download_top_players(num_players=3)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ Downloaded {len(files)} files total")
    logger.info(f"   Location: {output_dir}")
    logger.info("=" * 60)
    
    return files


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    # downloader = ChesscomDownloader()
    # downloader.download_player_games('nakamura')
    # downloader.download_top_players()
    pass
