"""
Language and translation system
"""

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
        
        # Map Selection
        'map_select_title': 'Select Map',
        'map_select': 'Select',
        'map_random': 'Random Map',
        'map_back': 'Back',
        
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
        
        # Map Selection
        'map_select_title': 'Sélectionner la Carte',
        'map_select': 'Sélectionner',
        'map_random': 'Carte Aléatoire',
        'map_back': 'Retour',
        
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
        
        # Map Selection
        'map_select_title': '맵 선택',
        'map_select': '선택',
        'map_random': '무작위 맵',
        'map_back': '뒤로',
        
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
        
        # Map Selection
        'map_select_title': 'Seleccionar Mapa',
        'map_select': 'Seleccionar',
        'map_random': 'Mapa Aleatorio',
        'map_back': 'Atrás',
        
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
    }
}

LANGUAGE_NAMES = {
    'english': 'English',
    'french': 'Français',
    'korean': '한국어',
    'spanish': 'Español'
}


class Language:
    """Language manager for translations."""
    
    def __init__(self, language='english'):
        """Initialize language manager."""
        self.current_language = language.lower()
        if self.current_language not in TRANSLATIONS:
            print(f"⚠️  Language '{language}' not found, using English")
            self.current_language = 'english'
    
    def set_language(self, language):
        """Set current language."""
        language = language.lower()
        if language in TRANSLATIONS:
            self.current_language = language
            print(f"✅ Language set to: {LANGUAGE_NAMES[language]}")
        else:
            print(f"❌ Language '{language}' not available")
    
    def get(self, key, default=None):
        """Get translation for key."""
        translations = TRANSLATIONS.get(self.current_language, TRANSLATIONS['english'])
        return translations.get(key, default or key)
    
    def get_all_languages(self):
        """Get list of all available languages."""
        return list(TRANSLATIONS.keys())
    
    def get_language_display_name(self, language=None):
        """Get display name for language."""
        if language is None:
            language = self.current_language
        return LANGUAGE_NAMES.get(language, language.capitalize())


# Global language instance
_language_instance = None

def get_language():
    """Get global language instance."""
    global _language_instance
    if _language_instance is None:
        from reinforcetactics.utils.settings import get_settings
        settings = get_settings()
        _language_instance = Language(settings.get_language())
    return _language_instance

def t(key, default=None):
    """Shorthand for translation."""
    return get_language().get(key, default)
