from dataclasses import dataclass, asdict

DAY = 17280  # одни сутки в единицах timestamp Yambda (бины по 5 секунд)


@dataclass
class Cfg:
    # данные
    data_dir: str | None = None   # локальная папка с parquet Yambda; если файла нет - скачаем с HF
    size: str = "50m"
    n_users: int = 2000           # None = все пользователи; история каждого сохраняется полностью
    seed: int = 0
    # временное разбиение: [train cutoffs] < [val cutoffs] < [test cutoffs]
    horizon_days: int = 1         # длина окна, в котором берём метки
    n_train_cutoffs: int = 6
    train_stride_days: int = 3    # шаг между train-отсечками (>= horizon_days)
    n_val_cutoffs: int = 3
    n_test_cutoffs: int = 3
    # цель и кандидаты
    pos_ratio_thr: int = 50       # положительный исход: прослушано >= 50% трека (Listen+)
    min_hist_events: int = 50     # пользователи с меньшей историей к моменту отсечки не берутся
    n_neg_pop: int = 20           # негативы из популярного за последние pop_window_days
    n_neg_hist: int = 20          # негативы из прошлой истории пользователя
    n_neg_art: int = 0            # "трудные" негативы: треки из топ-5 артистов пользователя за 30 дней
    pop_window_days: int = 7
    # прочее
    work_dir: str = "work"

    def tag(self) -> str:
        return (f"{self.size}_u{self.n_users}_s{self.seed}_h{self.horizon_days}"
                f"_t{self.n_train_cutoffs}x{self.train_stride_days}_v{self.n_val_cutoffs}_e{self.n_test_cutoffs}"
                f"_n{self.n_neg_pop}-{self.n_neg_hist}" + (f"-{self.n_neg_art}" if self.n_neg_art else ""))

    def dict(self):
        return asdict(self)
