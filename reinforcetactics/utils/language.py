"""
Language and translation system
"""
from typing import Optional

TRANSLATIONS = {
    'english': {
        # Main Menu
        'main_title': 'REINFORCE TACTICS',
        'main_subtitle': 'Turn-Based Strategy with RL',
        'menu_1v1_human': '1v1 (Human vs Human)',
        'menu_1v1_computer': '1v1 (Human vs Computer)',
        'menu_replay': 'Watch Replay',
        'menu_load': 'Load Game',
        'menu_settings': 'Settings',
        'menu_exit': 'Exit',
        'press_esc': 'Press ESC to exit',

        # Main Menu (alternative keys used in menus.py)
        'main_menu.title': 'Reinforce Tactics',
        'main_menu.new_game': 'New Game',
        'main_menu.load_game': 'Load Game',
        'main_menu.watch_replay': 'Watch Replay',
        'main_menu.credits': 'Credits',
        'main_menu.settings': 'Settings',
        'main_menu.quit': 'Quit',

        # Settings Menu
        'settings_title': 'SETTINGS',
        'settings_language': 'Language',
        'settings_paths': 'File Paths',
        'settings_video': 'Video Settings',
        'settings_maps_path': 'Maps Directory',
        'settings_videos_path': 'Videos Directory',
        'settings_replays_path': 'Replays Directory',
        'settings_saves_path': 'Saves Directory',
        'settings_reset': 'Reset to Defaults',
        'settings_back': 'Back',
        'settings_save': 'Save',
        'settings_saved': 'Settings saved!',
        'settings_reset_confirm': 'Reset all settings to defaults?',

        # Settings Menu (alternative keys)
        'settings.title': 'Settings',
        'settings.language': 'Language',
        'settings.sound': 'Sound',
        'settings.fullscreen': 'Fullscreen',

        # Map Selection
        'map_select_title': 'Select Map',
        'map_select': 'Select',
        'map_random': 'Random Map',
        'map_back': 'Back',

        # New Game Menu
        'new_game.title': 'Select Map',
        'new_game.select_mode': 'Select Game Mode',

        # Player Configuration Menu
        'player_config.title': 'Configure Players',
        'player_config.player': 'Player {number}',
        'player_config.type_human': 'Human',
        'player_config.type_computer': 'Computer',
        'player_config.difficulty': 'Difficulty',
        'player_config.difficulty_simple': 'SimpleBot',
        'player_config.difficulty_normal': 'NormalBot (Coming Soon)',
        'player_config.difficulty_hard': 'HardBot (Coming Soon)',
        'player_config.start_game': 'Start Game',

        # Save/Load Game
        'save_game.title': 'Save Game',
        'save_game.enter_name': 'Enter save name:',
        'save_game.instructions': 'Press ENTER to save, ESC to cancel',
        'load_game.title': 'Load Game',

        # Language Menu
        'language.title': 'Select Language',

        # Pause Menu
        'pause.title': 'Paused',
        'pause.resume': 'Resume',
        'pause.save': 'Save Game',
        'pause.load': 'Load Game',
        'pause.settings': 'Settings',
        'pause.main_menu': 'Main Menu',
        'pause.quit': 'Quit',

        # Game Over
        'game_over.title': 'Game Over',
        'game_over.winner': 'Player {player} Wins!',
        'game_over.save_replay': 'Save Replay',
        'game_over.new_game': 'New Game',
        'game_over.main_menu': 'Main Menu',
        'game_over.quit': 'Quit',

        # Common
        'common.back': 'Back',
        'common.confirm': 'Confirm',
        'common.cancel': 'Cancel',
        'common.esc_hint': 'Press ESC to go back',

        # Credits
        'credits.title': 'Credits',
        'credits.game_title': 'REINFORCE TACTICS',
        'credits.developer': 'Developer:',
        'credits.developer_name': 'Michael Kudlaty',
        'credits.description': 'Turn-Based Strategy Game with Reinforcement Learning',

        # Game
        'player': 'Player',
        'gold': 'Gold',
        'turn': 'Turn',
        'end_turn': 'End Turn',
        'resign': 'Resign',
        'game_over': 'Game Over!',
        'winner': 'wins!',
        'controls_title': 'Controls',
        'controls_select': 'Click units to select',
        'controls_move': 'Click tiles to move',
        'controls_end_turn': 'Press SPACE to end turn',
        'controls_quit': 'Press ESC to quit',

        # Units
        'warrior': 'Warrior',
        'mage': 'Mage',
        'cleric': 'Cleric',
        'barbarian': 'Barbarian',
        'health': 'Health',
        'attack': 'Attack',
        'movement': 'Movement',
        'cost': 'Cost',
        'status': 'Status',
        'abilities': 'Abilities',

        # Messages
        'loading': 'Loading...',
        'saving': 'Saving...',
        'not_implemented': 'Not yet implemented',
        'error': 'Error',
        'success': 'Success',
        'confirm': 'Confirm',
        'cancel': 'Cancel',

        # Map Editor
        'map_editor.title': 'Map Editor',
        'map_editor.new_map': 'New Map',
        'map_editor.edit_map': 'Edit Existing Map',
        'map_editor.save': 'Save Map',
        'map_editor.load': 'Load Map',
        'map_editor.new_map_dialog.title': 'Create New Map',
        'map_editor.new_map_dialog.width': 'Width:',
        'map_editor.new_map_dialog.height': 'Height:',
        'map_editor.new_map_dialog.players': 'Players:',
        'map_editor.new_map_dialog.create': 'Create',
        'map_editor.new_map_dialog.min_size': 'Minimum size: {size}x{size}',
        'map_editor.tile_palette.title': 'Tile Palette',
        'map_editor.tile_palette.terrain': 'Terrain',
        'map_editor.tile_palette.structures': 'Structures',
        'map_editor.tile_palette.neutral': 'Neutral',
        'map_editor.tile_palette.player': 'Player {number}',
        'map_editor.canvas.grid': 'Grid: On',
        'map_editor.canvas.grid_off': 'Grid: Off',
        'map_editor.canvas.coordinates': 'X: {x}, Y: {y}',
        'map_editor.tools.paint': 'Paint',
        'map_editor.tools.erase': 'Erase',
        'map_editor.tools.fill': 'Fill',
        'map_editor.save_dialog.title': 'Save Map',
        'map_editor.save_dialog.filename': 'Filename:',
        'map_editor.save_dialog.overwrite': 'File exists. Overwrite?',
        'map_editor.validation.no_hq': 'Each player must have exactly one Headquarters',
        'map_editor.validation.min_size': 'Map must be at least {size}x{size}',
        'map_editor.validation.invalid_hq': 'Player {player} needs exactly one Headquarters',
        'map_editor.shortcuts.title': 'Keyboard Shortcuts',
        'map_editor.shortcuts.save': 'Ctrl+S: Save',
        'map_editor.shortcuts.new': 'Ctrl+N: New Map',
        'map_editor.shortcuts.open': 'Ctrl+O: Open Map',
        'map_editor.shortcuts.grid': 'G: Toggle Grid',
        'map_editor.shortcuts.esc': 'Esc: Exit',
    },

    'french': {
        # Main Menu
        'main_title': 'REINFORCE TACTICS',
        'main_subtitle': 'Stratégie au tour par tour avec IA',
        'menu_1v1_human': '1v1 (Humain vs Humain)',
        'menu_1v1_computer': '1v1 (Humain vs Ordinateur)',
        'menu_replay': 'Regarder Replay',
        'menu_load': 'Charger Partie',
        'menu_settings': 'Paramètres',
        'menu_exit': 'Quitter',
        'press_esc': 'Appuyez sur ESC pour quitter',

        # Main Menu (alternative keys)
        'main_menu.title': 'Reinforce Tactics',
        'main_menu.new_game': 'Nouvelle Partie',
        'main_menu.load_game': 'Charger Partie',
        'main_menu.watch_replay': 'Regarder Replay',
        'main_menu.credits': 'Crédits',
        'main_menu.settings': 'Paramètres',
        'main_menu.quit': 'Quitter',

        # Settings Menu
        'settings_title': 'PARAMÈTRES',
        'settings_language': 'Langue',
        'settings_paths': 'Chemins des Fichiers',
        'settings_video': 'Paramètres Vidéo',
        'settings_maps_path': 'Répertoire des Cartes',
        'settings_videos_path': 'Répertoire des Vidéos',
        'settings_replays_path': 'Répertoire des Replays',
        'settings_saves_path': 'Répertoire des Sauvegardes',
        'settings_reset': 'Réinitialiser',
        'settings_back': 'Retour',
        'settings_save': 'Enregistrer',
        'settings_saved': 'Paramètres enregistrés!',
        'settings_reset_confirm': 'Réinitialiser tous les paramètres?',

        # Settings Menu (alternative keys)
        'settings.title': 'Paramètres',
        'settings.language': 'Langue',
        'settings.sound': 'Son',
        'settings.fullscreen': 'Plein Écran',

        # Map Selection
        'map_select_title': 'Sélectionner la Carte',
        'map_select': 'Sélectionner',
        'map_random': 'Carte Aléatoire',
        'map_back': 'Retour',

        # New Game Menu
        'new_game.title': 'Sélectionner la Carte',
        'new_game.select_mode': 'Choisir le Mode de Jeu',

        # Player Configuration Menu
        'player_config.title': 'Configurer les Joueurs',
        'player_config.player': 'Joueur {number}',
        'player_config.type_human': 'Humain',
        'player_config.type_computer': 'Ordinateur',
        'player_config.difficulty': 'Difficulté',
        'player_config.difficulty_simple': 'SimpleBot',
        'player_config.difficulty_normal': 'NormalBot (Bientôt)',
        'player_config.difficulty_hard': 'HardBot (Bientôt)',
        'player_config.start_game': 'Commencer',

        # Save/Load Game
        'save_game.title': 'Sauvegarder',
        'save_game.enter_name': 'Nom de la sauvegarde:',
        'save_game.instructions': 'Appuyez sur ENTRÉE pour sauvegarder, ESC pour annuler',
        'load_game.title': 'Charger Partie',

        # Language Menu
        'language.title': 'Choisir la Langue',

        # Pause Menu
        'pause.title': 'Pause',
        'pause.resume': 'Reprendre',
        'pause.save': 'Sauvegarder',
        'pause.load': 'Charger',
        'pause.settings': 'Paramètres',
        'pause.main_menu': 'Menu Principal',
        'pause.quit': 'Quitter',

        # Game Over
        'game_over.title': 'Partie Terminée',
        'game_over.winner': 'Joueur {player} Gagne!',
        'game_over.save_replay': 'Sauvegarder Replay',
        'game_over.new_game': 'Nouvelle Partie',
        'game_over.main_menu': 'Menu Principal',
        'game_over.quit': 'Quitter',

        # Common
        'common.back': 'Retour',
        'common.confirm': 'Confirmer',
        'common.cancel': 'Annuler',
        'common.esc_hint': 'Appuyez sur ESC pour revenir',

        # Credits
        'credits.title': 'Crédits',
        'credits.game_title': 'REINFORCE TACTICS',
        'credits.developer': 'Développeur:',
        'credits.developer_name': 'Michael Kudlaty',
        'credits.description': 'Jeu de Stratégie au Tour par Tour avec Apprentissage par Renforcement',

        # Game
        'player': 'Joueur',
        'gold': 'Or',
        'turn': 'Tour',
        'end_turn': 'Fin du Tour',
        'resign': 'Abandonner',
        'game_over': 'Partie Terminée!',
        'winner': 'gagne!',
        'controls_title': 'Contrôles',
        'controls_select': 'Cliquez pour sélectionner',
        'controls_move': 'Cliquez pour déplacer',
        'controls_end_turn': 'ESPACE pour finir le tour',
        'controls_quit': 'ESC pour quitter',

        # Units
        'warrior': 'Guerrier',
        'mage': 'Mage',
        'cleric': 'Clerc',
        'barbarian': 'Barbare',
        'health': 'Santé',
        'attack': 'Attaque',
        'movement': 'Mouvement',
        'cost': 'Coût',
        'status': 'Statut',
        'abilities': 'Capacités',

        # Messages
        'loading': 'Chargement...',
        'saving': 'Sauvegarde...',
        'not_implemented': 'Pas encore implémenté',
        'error': 'Erreur',
        'success': 'Succès',
        'confirm': 'Confirmer',
        'cancel': 'Annuler',

        # Map Editor
        'map_editor.title': 'Éditeur de Carte',
        'map_editor.new_map': 'Nouvelle Carte',
        'map_editor.edit_map': 'Modifier une Carte',
        'map_editor.save': 'Enregistrer Carte',
        'map_editor.load': 'Charger Carte',
        'map_editor.new_map_dialog.title': 'Créer une Nouvelle Carte',
        'map_editor.new_map_dialog.width': 'Largeur:',
        'map_editor.new_map_dialog.height': 'Hauteur:',
        'map_editor.new_map_dialog.players': 'Joueurs:',
        'map_editor.new_map_dialog.create': 'Créer',
        'map_editor.new_map_dialog.min_size': 'Taille minimum: {size}x{size}',
        'map_editor.tile_palette.title': 'Palette de Tuiles',
        'map_editor.tile_palette.terrain': 'Terrain',
        'map_editor.tile_palette.structures': 'Structures',
        'map_editor.tile_palette.neutral': 'Neutre',
        'map_editor.tile_palette.player': 'Joueur {number}',
        'map_editor.canvas.grid': 'Grille: Activée',
        'map_editor.canvas.grid_off': 'Grille: Désactivée',
        'map_editor.canvas.coordinates': 'X: {x}, Y: {y}',
        'map_editor.tools.paint': 'Pinceau',
        'map_editor.tools.erase': 'Effacer',
        'map_editor.tools.fill': 'Remplir',
        'map_editor.save_dialog.title': 'Enregistrer Carte',
        'map_editor.save_dialog.filename': 'Nom de fichier:',
        'map_editor.save_dialog.overwrite': 'Le fichier existe. Écraser?',
        'map_editor.validation.no_hq': 'Chaque joueur doit avoir exactement un Quartier Général',
        'map_editor.validation.min_size': 'La carte doit faire au moins {size}x{size}',
        'map_editor.validation.invalid_hq': 'Le joueur {player} a besoin d\'un Quartier Général',
        'map_editor.shortcuts.title': 'Raccourcis Clavier',
        'map_editor.shortcuts.save': 'Ctrl+S: Enregistrer',
        'map_editor.shortcuts.new': 'Ctrl+N: Nouvelle Carte',
        'map_editor.shortcuts.open': 'Ctrl+O: Ouvrir Carte',
        'map_editor.shortcuts.grid': 'G: Activer/Désactiver Grille',
        'map_editor.shortcuts.esc': 'Esc: Quitter',
    },

    'korean': {
        # Main Menu
        'main_title': 'REINFORCE TACTICS',
        'main_subtitle': '턴제 전략 게임 with RL',
        'menu_1v1_human': '1v1 (인간 vs 인간)',
        'menu_1v1_computer': '1v1 (인간 vs 컴퓨터)',
        'menu_replay': '리플레이 보기',
        'menu_load': '게임 불러오기',
        'menu_settings': '설정',
        'menu_exit': '종료',
        'press_esc': 'ESC를 눌러 종료',

        # Main Menu (alternative keys)
        'main_menu.title': 'Reinforce Tactics',
        'main_menu.new_game': '새 게임',
        'main_menu.load_game': '불러오기',
        'main_menu.watch_replay': '리플레이 보기',
        'main_menu.credits': '크레딧',
        'main_menu.settings': '설정',
        'main_menu.quit': '종료',

        # Settings Menu
        'settings_title': '설정',
        'settings_language': '언어',
        'settings_paths': '파일 경로',
        'settings_video': '비디오 설정',
        'settings_maps_path': '맵 디렉토리',
        'settings_videos_path': '비디오 디렉토리',
        'settings_replays_path': '리플레이 디렉토리',
        'settings_saves_path': '저장 디렉토리',
        'settings_reset': '기본값으로 재설정',
        'settings_back': '뒤로',
        'settings_save': '저장',
        'settings_saved': '설정이 저장되었습니다!',
        'settings_reset_confirm': '모든 설정을 기본값으로 재설정하시겠습니까?',

        # Settings Menu (alternative keys)
        'settings.title': '설정',
        'settings.language': '언어',
        'settings.sound': '소리',
        'settings.fullscreen': '전체 화면',

        # Map Selection
        'map_select_title': '맵 선택',
        'map_select': '선택',
        'map_random': '무작위 맵',
        'map_back': '뒤로',

        # New Game Menu
        'new_game.title': '맵 선택',
        'new_game.select_mode': '게임 모드 선택',

        # Player Configuration Menu
        'player_config.title': '플레이어 설정',
        'player_config.player': '플레이어 {number}',
        'player_config.type_human': '인간',
        'player_config.type_computer': '컴퓨터',
        'player_config.difficulty': '난이도',
        'player_config.difficulty_simple': 'SimpleBot',
        'player_config.difficulty_normal': 'NormalBot (곧 출시)',
        'player_config.difficulty_hard': 'HardBot (곧 출시)',
        'player_config.start_game': '게임 시작',

        # Save/Load Game
        'save_game.title': '게임 저장',
        'save_game.enter_name': '저장 이름 입력:',
        'save_game.instructions': 'ENTER로 저장, ESC로 취소',
        'load_game.title': '게임 불러오기',

        # Language Menu
        'language.title': '언어 선택',

        # Pause Menu
        'pause.title': '일시 정지',
        'pause.resume': '계속',
        'pause.save': '저장',
        'pause.load': '불러오기',
        'pause.settings': '설정',
        'pause.main_menu': '메인 메뉴',
        'pause.quit': '종료',

        # Game Over
        'game_over.title': '게임 종료',
        'game_over.winner': '플레이어 {player} 승리!',
        'game_over.save_replay': '리플레이 저장',
        'game_over.new_game': '새 게임',
        'game_over.main_menu': '메인 메뉴',
        'game_over.quit': '종료',

        # Common
        'common.back': '뒤로',
        'common.confirm': '확인',
        'common.cancel': '취소',
        'common.esc_hint': 'ESC를 눌러 뒤로',

        # Credits
        'credits.title': '크레딧',
        'credits.game_title': 'REINFORCE TACTICS',
        'credits.developer': '개발자:',
        'credits.developer_name': 'Michael Kudlaty',
        'credits.description': '강화 학습 기반 턴제 전략 게임',

        # Game
        'player': '플레이어',
        'gold': '골드',
        'turn': '턴',
        'end_turn': '턴 종료',
        'resign': '포기',
        'game_over': '게임 종료!',
        'winner': '승리!',
        'controls_title': '조작법',
        'controls_select': '유닛을 클릭하여 선택',
        'controls_move': '타일을 클릭하여 이동',
        'controls_end_turn': 'SPACE로 턴 종료',
        'controls_quit': 'ESC로 종료',

        # Units
        'warrior': '전사',
        'mage': '마법사',
        'cleric': '성직자',
        'barbarian': '야만인',
        'health': '체력',
        'attack': '공격력',
        'movement': '이동력',
        'cost': '비용',
        'status': '상태',
        'abilities': '능력',

        # Messages
        'loading': '로딩 중...',
        'saving': '저장 중...',
        'not_implemented': '아직 구현되지 않음',
        'error': '오류',
        'success': '성공',
        'confirm': '확인',
        'cancel': '취소',

        # Map Editor
        'map_editor.title': '맵 에디터',
        'map_editor.new_map': '새 맵',
        'map_editor.edit_map': '기존 맵 수정',
        'map_editor.save': '맵 저장',
        'map_editor.load': '맵 불러오기',
        'map_editor.new_map_dialog.title': '새 맵 만들기',
        'map_editor.new_map_dialog.width': '가로:',
        'map_editor.new_map_dialog.height': '세로:',
        'map_editor.new_map_dialog.players': '플레이어:',
        'map_editor.new_map_dialog.create': '생성',
        'map_editor.new_map_dialog.min_size': '최소 크기: {size}x{size}',
        'map_editor.tile_palette.title': '타일 팔레트',
        'map_editor.tile_palette.terrain': '지형',
        'map_editor.tile_palette.structures': '구조물',
        'map_editor.tile_palette.neutral': '중립',
        'map_editor.tile_palette.player': '플레이어 {number}',
        'map_editor.canvas.grid': '그리드: 켜짐',
        'map_editor.canvas.grid_off': '그리드: 꺼짐',
        'map_editor.canvas.coordinates': 'X: {x}, Y: {y}',
        'map_editor.tools.paint': '그리기',
        'map_editor.tools.erase': '지우기',
        'map_editor.tools.fill': '채우기',
        'map_editor.save_dialog.title': '맵 저장',
        'map_editor.save_dialog.filename': '파일 이름:',
        'map_editor.save_dialog.overwrite': '파일이 존재합니다. 덮어쓰시겠습니까?',
        'map_editor.validation.no_hq': '각 플레이어는 정확히 하나의 본부가 있어야 합니다',
        'map_editor.validation.min_size': '맵은 최소 {size}x{size}여야 합니다',
        'map_editor.validation.invalid_hq': '플레이어 {player}에게 정확히 하나의 본부가 필요합니다',
        'map_editor.shortcuts.title': '단축키',
        'map_editor.shortcuts.save': 'Ctrl+S: 저장',
        'map_editor.shortcuts.new': 'Ctrl+N: 새 맵',
        'map_editor.shortcuts.open': 'Ctrl+O: 맵 열기',
        'map_editor.shortcuts.grid': 'G: 그리드 토글',
        'map_editor.shortcuts.esc': 'Esc: 나가기',
    },

    'spanish': {
        # Main Menu
        'main_title': 'REINFORCE TACTICS',
        'main_subtitle': 'Estrategia por Turnos con IA',
        'menu_1v1_human': '1v1 (Humano vs Humano)',
        'menu_1v1_computer': '1v1 (Humano vs Computadora)',
        'menu_replay': 'Ver Repetición',
        'menu_load': 'Cargar Juego',
        'menu_settings': 'Configuración',
        'menu_exit': 'Salir',
        'press_esc': 'Presiona ESC para salir',

        # Main Menu (alternative keys)
        'main_menu.title': 'Reinforce Tactics',
        'main_menu.new_game': 'Nuevo Juego',
        'main_menu.load_game': 'Cargar Juego',
        'main_menu.watch_replay': 'Ver Repetición',
        'main_menu.credits': 'Créditos',
        'main_menu.settings': 'Configuración',
        'main_menu.quit': 'Salir',

        # Settings Menu
        'settings_title': 'CONFIGURACIÓN',
        'settings_language': 'Idioma',
        'settings_paths': 'Rutas de Archivos',
        'settings_video': 'Configuración de Video',
        'settings_maps_path': 'Directorio de Mapas',
        'settings_videos_path': 'Directorio de Videos',
        'settings_replays_path': 'Directorio de Repeticiones',
        'settings_saves_path': 'Directorio de Guardados',
        'settings_reset': 'Restablecer Valores',
        'settings_back': 'Atrás',
        'settings_save': 'Guardar',
        'settings_saved': '¡Configuración guardada!',
        'settings_reset_confirm': '¿Restablecer toda la configuración?',

        # Settings Menu (alternative keys)
        'settings.title': 'Configuración',
        'settings.language': 'Idioma',
        'settings.sound': 'Sonido',
        'settings.fullscreen': 'Pantalla Completa',

        # Map Selection
        'map_select_title': 'Seleccionar Mapa',
        'map_select': 'Seleccionar',
        'map_random': 'Mapa Aleatorio',
        'map_back': 'Atrás',

        # New Game Menu
        'new_game.title': 'Seleccionar Mapa',
        'new_game.select_mode': 'Seleccionar Modo de Juego',

        # Player Configuration Menu
        'player_config.title': 'Configurar Jugadores',
        'player_config.player': 'Jugador {number}',
        'player_config.type_human': 'Humano',
        'player_config.type_computer': 'Computadora',
        'player_config.difficulty': 'Dificultad',
        'player_config.difficulty_simple': 'SimpleBot',
        'player_config.difficulty_normal': 'NormalBot (Próximamente)',
        'player_config.difficulty_hard': 'HardBot (Próximamente)',
        'player_config.start_game': 'Comenzar Juego',

        # Save/Load Game
        'save_game.title': 'Guardar Juego',
        'save_game.enter_name': 'Nombre del guardado:',
        'save_game.instructions': 'Presiona ENTER para guardar, ESC para cancelar',
        'load_game.title': 'Cargar Juego',

        # Language Menu
        'language.title': 'Seleccionar Idioma',

        # Pause Menu
        'pause.title': 'Pausa',
        'pause.resume': 'Continuar',
        'pause.save': 'Guardar',
        'pause.load': 'Cargar',
        'pause.settings': 'Configuración',
        'pause.main_menu': 'Menú Principal',
        'pause.quit': 'Salir',

        # Game Over
        'game_over.title': 'Fin del Juego',
        'game_over.winner': '¡Jugador {player} Gana!',
        'game_over.save_replay': 'Guardar Repetición',
        'game_over.new_game': 'Nuevo Juego',
        'game_over.main_menu': 'Menú Principal',
        'game_over.quit': 'Salir',

        # Common
        'common.back': 'Atrás',
        'common.confirm': 'Confirmar',
        'common.cancel': 'Cancelar',
        'common.esc_hint': 'Presiona ESC para volver',

        # Credits
        'credits.title': 'Créditos',
        'credits.game_title': 'REINFORCE TACTICS',
        'credits.developer': 'Desarrollador:',
        'credits.developer_name': 'Michael Kudlaty',
        'credits.description': 'Juego de Estrategia por Turnos con Aprendizaje por Refuerzo',

        # Game
        'player': 'Jugador',
        'gold': 'Oro',
        'turn': 'Turno',
        'end_turn': 'Fin de Turno',
        'resign': 'Rendirse',
        'game_over': '¡Juego Terminado!',
        'winner': '¡gana!',
        'controls_title': 'Controles',
        'controls_select': 'Clic para seleccionar unidades',
        'controls_move': 'Clic para mover',
        'controls_end_turn': 'ESPACIO para terminar turno',
        'controls_quit': 'ESC para salir',

        # Units
        'warrior': 'Guerrero',
        'mage': 'Mago',
        'cleric': 'Clérigo',
        'barbarian': 'Bárbaro',
        'health': 'Salud',
        'attack': 'Ataque',
        'movement': 'Movimiento',
        'cost': 'Costo',
        'status': 'Estado',
        'abilities': 'Habilidades',

        # Messages
        'loading': 'Cargando...',
        'saving': 'Guardando...',
        'not_implemented': 'No implementado aún',
        'error': 'Error',
        'success': 'Éxito',
        'confirm': 'Confirmar',
        'cancel': 'Cancelar',
    },

    'chinese': {
        # Main Menu
        'main_title': 'REINFORCE TACTICS',
        'main_subtitle': '回合制策略游戏 with RL',
        'menu_1v1_human': '1v1 (人类 vs 人类)',
        'menu_1v1_computer': '1v1 (人类 vs 电脑)',
        'menu_replay': '观看回放',
        'menu_load': '加载游戏',
        'menu_settings': '设置',
        'menu_exit': '退出',
        'press_esc': '按ESC退出',

        # Main Menu (alternative keys)
        'main_menu.title': 'Reinforce Tactics',
        'main_menu.new_game': '新游戏',
        'main_menu.load_game': '加载游戏',
        'main_menu.watch_replay': '观看回放',
        'main_menu.credits': '制作人员',
        'main_menu.settings': '设置',
        'main_menu.quit': '退出',

        # Settings Menu
        'settings_title': '设置',
        'settings_language': '语言',
        'settings_paths': '文件路径',
        'settings_video': '视频设置',
        'settings_maps_path': '地图目录',
        'settings_videos_path': '视频目录',
        'settings_replays_path': '回放目录',
        'settings_saves_path': '存档目录',
        'settings_reset': '恢复默认',
        'settings_back': '返回',
        'settings_save': '保存',
        'settings_saved': '设置已保存！',
        'settings_reset_confirm': '确定要恢复默认设置吗？',

        # Settings Menu (alternative keys)
        'settings.title': '设置',
        'settings.language': '语言',
        'settings.sound': '声音',
        'settings.fullscreen': '全屏',

        # Map Selection
        'map_select_title': '选择地图',
        'map_select': '选择',
        'map_random': '随机地图',
        'map_back': '返回',

        # New Game Menu
        'new_game.title': '选择地图',
        'new_game.select_mode': '选择游戏模式',

        # Player Configuration Menu
        'player_config.title': '配置玩家',
        'player_config.player': '玩家 {number}',
        'player_config.type_human': '人类',
        'player_config.type_computer': '电脑',
        'player_config.difficulty': '难度',
        'player_config.difficulty_simple': 'SimpleBot',
        'player_config.difficulty_normal': 'NormalBot (即将推出)',
        'player_config.difficulty_hard': 'HardBot (即将推出)',
        'player_config.start_game': '开始游戏',

        # Save/Load Game
        'save_game.title': '保存游戏',
        'save_game.enter_name': '输入存档名称：',
        'save_game.instructions': '按回车保存，按ESC取消',
        'load_game.title': '加载游戏',

        # Language Menu
        'language.title': '选择语言',

        # Pause Menu
        'pause.title': '暂停',
        'pause.resume': '继续',
        'pause.save': '保存',
        'pause.load': '加载',
        'pause.settings': '设置',
        'pause.main_menu': '主菜单',
        'pause.quit': '退出',

        # Game Over
        'game_over.title': '游戏结束',
        'game_over.winner': '玩家 {player} 获胜！',
        'game_over.save_replay': '保存回放',
        'game_over.new_game': '新游戏',
        'game_over.main_menu': '主菜单',
        'game_over.quit': '退出',

        # Common
        'common.back': '返回',
        'common.confirm': '确认',
        'common.cancel': '取消',
        'common.esc_hint': '按ESC返回',

        # Credits
        'credits.title': '制作人员',
        'credits.game_title': 'REINFORCE TACTICS',
        'credits.developer': '开发者:',
        'credits.developer_name': 'Michael Kudlaty',
        'credits.description': '基于强化学习的回合制策略游戏',

        # Game
        'player': '玩家',
        'gold': '金币',
        'turn': '回合',
        'end_turn': '结束回合',
        'resign': '投降',
        'game_over': '游戏结束！',
        'winner': '获胜！',
        'controls_title': '操作说明',
        'controls_select': '点击选择单位',
        'controls_move': '点击移动',
        'controls_end_turn': '空格键结束回合',
        'controls_quit': 'ESC退出',

        # Units
        'warrior': '战士',
        'mage': '法师',
        'cleric': '牧师',
        'barbarian': '野蛮人',
        'health': '生命值',
        'attack': '攻击力',
        'movement': '移动力',
        'cost': '费用',
        'status': '状态',
        'abilities': '技能',

        # Messages
        'loading': '加载中...',
        'saving': '保存中...',
        'not_implemented': '尚未实现',
        'error': '错误',
        'success': '成功',
        'confirm': '确认',
        'cancel': '取消',
    }
}

LANGUAGE_NAMES = {
    'english': 'English',
    'french': 'Français',
    'korean': '한국어',
    'spanish': 'Español',
    'chinese': '中文'
}

# Language code mappings (for flexibility)
LANGUAGE_CODES = {
    'en': 'english',
    'fr': 'french',
    'ko': 'korean',
    'es': 'spanish',
    'zh': 'chinese',
    'english': 'english',
    'french': 'french',
    'korean': 'korean',
    'spanish': 'spanish',
    'chinese': 'chinese',
}


class Language:
    """Language manager for translations."""

    def __init__(self, language: str = 'english'):
        """
        Initialize language manager.

        Args:
            language: Language name or code (e.g., 'english', 'en', 'french', 'fr')
        """
        self.set_language(language)

    def set_language(self, language: str) -> bool:
        """
        Set current language.

        Args:
            language: Language name or code

        Returns:
            True if language was set successfully
        """
        # Normalize language code
        lang_key = language.lower()
        normalized = LANGUAGE_CODES.get(lang_key, lang_key)

        if normalized in TRANSLATIONS:
            self.current_language = normalized
            print(f"✅ Language set to: {LANGUAGE_NAMES.get(normalized, normalized)}")
            return True
        print(f"❌ Language '{language}' not available, using English")
        self.current_language = 'english'
        return False

    def get(self, key: str, default: Optional[str] = None) -> str:
        """
        Get translation for key.

        Args:
            key: Translation key (e.g., 'main_menu.title')
            default: Default value if key not found

        Returns:
            Translated string
        """
        translations = TRANSLATIONS.get(self.current_language, TRANSLATIONS['english'])
        result = translations.get(key)

        # If not found in current language, try English as fallback
        if result is None and self.current_language != 'english':
            result = TRANSLATIONS['english'].get(key)

        return result if result is not None else (default or key)

    def get_all_languages(self) -> list:
        """Get list of all available languages."""
        return list(TRANSLATIONS.keys())

    def get_language_display_name(self, language: Optional[str] = None) -> str:
        """
        Get display name for language.

        Args:
            language: Language code (uses current if None)

        Returns:
            Display name (e.g., 'English', 'Français')
        """
        if language is None:
            language = self.current_language

        # Normalize
        normalized = LANGUAGE_CODES.get(language.lower(), language.lower())
        return LANGUAGE_NAMES.get(normalized, language.capitalize())

    def get_current_language(self) -> str:
        """Get current language code."""
        return self.current_language


# Global language instance
_language_instance: Optional[Language] = None  # pylint: disable=invalid-name


def get_language() -> Language:
    """
    Get global language instance.

    Creates instance on first call using settings if available.

    Returns:
        Language instance
    """
    global _language_instance
    if _language_instance is None:
        # Try to get language from settings
        try:
            from reinforcetactics.utils.settings import get_settings
            settings = get_settings()
            _language_instance = Language(settings.get_language())
        except (ImportError, Exception):
            # Fallback to English if settings not available
            _language_instance = Language('english')
    return _language_instance


def reset_language(lang_code: str = 'english') -> Language:
    """
    Reset global language instance to a new language.

    This function was missing and is needed by menus.py LanguageMenu._set_language()

    Args:
        lang_code: Language code or name (e.g., 'en', 'english', 'fr', 'french')

    Returns:
        The new Language instance
    """
    global _language_instance
    _language_instance = Language(lang_code)

    # Also try to persist to settings
    try:
        from reinforcetactics.utils.settings import get_settings
        settings = get_settings()
        # Normalize the language code
        normalized = LANGUAGE_CODES.get(lang_code.lower(), lang_code.lower())
        settings.set_language(normalized)
    except (ImportError, Exception):
        pass  # Settings not available, just update in-memory

    return _language_instance


def t(key: str, default: Optional[str] = None) -> str:
    """
    Shorthand for translation.

    Args:
        key: Translation key
        default: Default value if not found

    Returns:
        Translated string
    """
    return get_language().get(key, default)
