mod annealers;
mod observer;
mod puzzmo;
mod words;

use clap::{Parser as _, ValueEnum as _};
use rand::SeedableRng as _;
use rayon::iter::{IntoParallelIterator as _, ParallelIterator as _};

type Tower = ndarray::Array2<char>;

const EIGHT_NEIGHBORS: &[(isize, isize)] = &[
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

fn find_paths(tower: &Tower, root: &words::Node) -> Vec<Vec<(usize, usize)>> {
    fn helper(
        tower: &Tower,
        path: &[(usize, usize)],
        node: &words::Node,
    ) -> Vec<Vec<(usize, usize)>> {
        let mut paths = vec![];
        if node.is_end() {
            paths.push(path.to_vec());
        }

        let (oi, oj) = path.last().unwrap();

        for &(di, dj) in EIGHT_NEIGHBORS {
            let Some(i) = oi.checked_add_signed(di) else {
                continue;
            };
            let Some(j) = oj.checked_add_signed(dj) else {
                continue;
            };
            if path.contains(&(i, j)) {
                continue;
            }
            let Some(&letter) = tower.get([i, j]) else {
                continue;
            };
            let Some(child) = node.get(letter.to_ascii_uppercase()) else {
                continue;
            };

            paths.extend(helper(
                tower,
                &path
                    .iter()
                    .cloned()
                    .chain(std::iter::once((i, j)))
                    .collect::<Vec<_>>(),
                child,
            ));
        }

        paths
    }

    let (n, m) = tower.dim();

    (0..n)
        .into_par_iter()
        .flat_map(|i| (0..m).into_par_iter().map(move |j| (i, j)))
        .flat_map(|(i, j)| {
            let Some(child) = root.get(tower[[i, j]].to_ascii_uppercase()) else {
                return vec![];
            };
            helper(tower, &[(i, j)], child)
        })
        .collect::<Vec<_>>()
}

fn score_letter(c: char) -> usize {
    match c {
        'A' => 1,
        'B' => 4,
        'C' => 4,
        'D' => 3,
        'E' => 1,
        'F' => 5,
        'G' => 3,
        'H' => 5,
        'I' => 1,
        'J' => 9,
        'K' => 6,
        'L' => 2,
        'M' => 4,
        'N' => 2,
        'O' => 1,
        'P' => 4,
        'Q' => 12,
        'R' => 2,
        'S' => 1,
        'T' => 2,
        'U' => 1,
        'V' => 5,
        'W' => 5,
        'X' => 9,
        'Y' => 5,
        'Z' => 11,
        _ => 0,
    }
}

fn deletable(tower: &Tower, path: &[(usize, usize)]) -> std::collections::HashSet<(usize, usize)> {
    let (_, m) = tower.dim();

    let mut collected = path
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>();

    for &(i, j) in path.iter() {
        let c = tower[[i, j]].to_ascii_uppercase();
        if !matches!(c, 'J' | 'Q' | 'X' | 'Z') {
            continue;
        }
        collected.extend(
            (0..m)
                .map(|j| (i, j))
                .filter(|&(i, j)| tower[[i, j]] != '\0'),
        );
    }

    for (oi, oj) in path.iter() {
        for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let Some(i) = oi.checked_add_signed(di) else {
                continue;
            };
            let Some(j) = oj.checked_add_signed(dj) else {
                continue;
            };
            let Some(&letter) = tower.get([i, j]) else {
                continue;
            };
            if letter == '\0' {
                continue;
            }
            if path.len() >= 5 || letter == '_' {
                collected.insert((i, j));
            }
        }
    }

    collected
}

fn delete_path(tower: &mut Tower, path: &[(usize, usize)]) {
    let (n, m) = tower.dim();

    for (i, j) in deletable(tower, path) {
        tower[[i, j]] = '\0';
    }

    for j in 0..m {
        let mut i2 = n - 1;

        'top: for i in (0..n).rev() {
            let Some(next_i2) = i2.checked_sub(1) else {
                break 'top;
            };
            i2 = next_i2;

            if tower[(i, j)] == '\0' {
                while tower[(i2, j)] == '\0' {
                    let Some(next_i2) = i2.checked_sub(1) else {
                        break 'top;
                    };
                    i2 = next_i2;
                }
                tower[(i, j)] = tower[(i2, j)];
                tower[(i2, j)] = '\0';
            }
        }
    }
}

fn score_path(tower: &Tower, path: &[(usize, usize)]) -> usize {
    deletable(tower, &path)
        .iter()
        .map(|&(i, j)| score_letter(tower[[i, j]].to_ascii_uppercase()))
        .sum::<usize>()
        * path.len()
        * (path
            .iter()
            .filter(|&&(i, j)| tower[[i, j]].is_ascii_lowercase())
            .count()
            + 1)
}

fn is_almost_there(tower: &Tower) -> bool {
    tower
        .columns()
        .into_iter()
        .all(|row| row.into_iter().filter(|&&x| x != '\0').count() <= 2)
}

fn score_solution(tower: &Tower, solution: &[impl AsRef<[(usize, usize)]>]) -> usize {
    let mut tower = tower.clone();

    let mut total_score = 0;
    for path in solution {
        let score = score_path(&tower, path.as_ref());
        total_score += score;
        delete_path(&mut tower, path.as_ref());
    }

    if is_almost_there(&tower) {
        total_score += 1000;
    }

    if tower.iter().filter(|&&x| x == '\0').count() == tower.len() {
        total_score += 1000;
    }

    total_score
}

fn nudge_solution(
    tower: &Tower,
    root: &words::Node,
    solution: &mut Vec<Vec<(usize, usize)>>,
    rng: &mut impl rand::Rng,
) {
    solution.drain(rng.gen_range(0..solution.len())..);

    let mut tower = tower.clone();
    for path in solution.iter() {
        delete_path(&mut tower, path);
    }

    let mut next_paths = find_paths(&tower, root);
    let path = next_paths.swap_remove(rng.gen_range(0..next_paths.len()));
    delete_path(&mut tower, &path);
    solution.push(path);

    loop {
        let Some(best) = find_paths(&tower, root)
            .into_iter()
            .max_by_key(|path| score_path(&tower, &path))
        else {
            break;
        };
        delete_path(&mut tower, &best);
        solution.push(best);
    }
}

fn solve_greedy(tower: &Tower, root: &words::Node) -> Vec<Vec<(usize, usize)>> {
    let mut tower = tower.clone();
    let mut solution = vec![];

    loop {
        let Some(best) = find_paths(&tower, root)
            .into_iter()
            .max_by_key(|path| score_path(&tower, &path))
        else {
            break;
        };
        delete_path(&mut tower, &best);
        solution.push(best);
    }

    solution
}

fn pretty_tower(tower: &Tower, path: &[(usize, usize)]) -> String {
    let (n, m) = tower.dim();

    #[derive(Clone, Copy, PartialEq)]
    enum LinkType {
        None,
        Vertical,
        Horizontal,
        Diagonal,
        Antidiagonal,
        Cross,
    }

    let mut links = ndarray::Array2::from_elem((n * 2 + 1, m * 2 + 1), LinkType::None);
    let deletable: std::collections::HashSet<(usize, usize)> = deletable(tower, path);

    if !path.is_empty() {
        for (&(ia, ja), &(ib, jb)) in path.iter().zip(path[1..].iter()) {
            let li = ia * 2 + 1;
            let lj = ja * 2 + 1;

            let di = ib as isize - ia as isize;
            let dj = jb as isize - ja as isize;

            let l = &mut links[[(li as isize + di) as usize, (lj as isize + dj) as usize]];
            match (di, dj) {
                (-1, 0) | (1, 0) => {
                    *l = LinkType::Vertical;
                }
                (0, -1) | (0, 1) => {
                    *l = LinkType::Horizontal;
                }
                (-1, -1) | (1, 1) => {
                    *l = if *l != LinkType::Antidiagonal {
                        LinkType::Diagonal
                    } else {
                        LinkType::Cross
                    };
                }
                (1, -1) | (-1, 1) => {
                    *l = if *l != LinkType::Diagonal {
                        LinkType::Antidiagonal
                    } else {
                        LinkType::Cross
                    };
                }
                _ => unreachable!(),
            }
        }
    }

    let mut pretty = ndarray::Array2::from_elem((n * 2 + 1, m * 2 + 1), " ".to_string());

    for (i, j) in (0..n).flat_map(|i| (0..m).map(move |j| (i, j))) {
        let pi = i * 2 + 1;
        let pj = j * 2 + 1;

        for &(di, dj) in EIGHT_NEIGHBORS {
            let li = (pi as isize + di) as usize;
            let lj = (pj as isize + dj) as usize;

            let link = links[[li, lj]];
            pretty[[li, lj]] = match link {
                LinkType::None => {
                    if dj == 0 {
                        "   "
                    } else {
                        " "
                    }
                }
                LinkType::Vertical => "\x1b[1;35m │ \x1b[0m",
                LinkType::Horizontal => "\x1b[1;35m─\x1b[0m",
                LinkType::Diagonal => "\x1b[1;35m╲\x1b[0m",
                LinkType::Antidiagonal => "\x1b[1;35m╱\x1b[0m",
                LinkType::Cross => "\x1b[1;35m╳\x1b[0m",
            }
            .to_string();
        }

        let c = match tower[[i, j]] {
            '\0' => ' ',
            '_' => '░',
            v => v,
        };
        pretty[[pi, pj]] = if path.first() == Some(&(i, j)) {
            format!("\x1b[1;37;45m {c} \x1b[0m")
        } else if path.contains(&(i, j)) {
            format!("\x1b[37;45m {c} \x1b[0m")
        } else if deletable.contains(&(i, j)) {
            format!("\x1b[35m {c} \x1b[0m")
        } else {
            format!(" {c} ")
        };
    }

    let word = path.iter().map(|&(i, j)| tower[[i, j]]).collect::<String>();
    let score = score_path(&tower, &path);

    let border_length = (m - 1) + m * 3;

    let bottom_border = std::iter::repeat("═")
        .take(border_length)
        .collect::<String>();

    let top_border = if !path.is_empty() {
        let header = format!("{word:} ({score:})");
        format!("═{header:═<width$}", width = border_length - 1)
    } else {
        bottom_border.clone()
    };

    let body = pretty
        .slice(ndarray::s![1..n * 2, 1..m * 2])
        .rows()
        .into_iter()
        .map(|cols| {
            format!(
                "║{}║",
                cols.into_iter().flat_map(|v| v.chars()).collect::<String>()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!("╔{top_border}╗\n{body}\n╚{bottom_border}╝")
}

#[derive(clap::ValueEnum, Clone)]
enum Coster {
    TotalScore,
    LongestWord,
    BestWord,
    FewestWords,
}

impl Coster {
    pub fn as_coster(&self) -> &'static annealers::Coster {
        match self {
            Coster::LongestWord => &annealers::LONGEST_WORD,
            Coster::TotalScore => &annealers::TOTAL_SCORE,
            Coster::BestWord => &annealers::BEST_WORD,
            Coster::FewestWords => &annealers::FEWEST_WORDS,
        }
    }
}

#[derive(clap::Parser)]
struct Args {
    #[arg(long, default_value = "total-score")]
    coster: Coster,

    #[arg(long)]
    day: Option<String>,

    #[arg(long)]
    status: Option<String>,

    #[arg(long)]
    allow_leftovers: bool,

    #[arg(long, default_value_t = 1000.0)]
    initial_temperature: f64,

    #[arg(long, default_value = "spelltower")]
    game: String,

    #[arg(long, default_value_t = 5000)]
    reannealing_fixed: u64,
}

fn parse_puzzle(p: &str) -> anyhow::Result<crate::Tower> {
    let mut puzzle_iter = p.split("\n");
    let _ = puzzle_iter
        .next()
        .ok_or_else(|| anyhow::anyhow!("no puzzle lines"))?;

    let dim = puzzle_iter
        .next()
        .ok_or_else(|| anyhow::anyhow!("no puzzle lines"))?;

    let (w, h) = dim
        .split_once("x")
        .ok_or_else(|| anyhow::anyhow!("invalid dimensions: {dim}"))?;

    let w: usize = w.parse()?;
    let h: usize = h.parse()?;

    Ok(ndarray::Array2::from_shape_vec(
        (h, w),
        puzzle_iter
            .flat_map(|row| row.chars())
            .take(w * h)
            .collect(),
    )?)
}

fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = Args::parse();

    let (words, _) = words::load();

    let puzzle = puzzmo::get_puzzle(
        &args.game,
        &args.status.as_ref().map(|v| v.as_str()).unwrap_or_else(|| {
            if args.game == "cubeclear" {
                "Experimental"
            } else {
                "Vanilla"
            }
        }),
        args.day,
    )?;

    log::info!(day = puzzle.day.as_str(), is_today = puzzle.is_today, coster = args.coster.to_possible_value().unwrap().get_name(); "spelltower solver");

    let tower = parse_puzzle(&puzzle.puzzle)?;
    println!("{}", pretty_tower(&tower, &[]));

    let solver =
        argmin::solver::simulatedannealing::SimulatedAnnealing::new(args.initial_temperature)?
            .with_reannealing_fixed(args.reannealing_fixed);

    let coster = args.coster.as_coster();
    let res = argmin::core::Executor::new(
        annealers::Annealer::new(
            &tower,
            &words,
            args.allow_leftovers,
            rand_xoshiro::Xoshiro256PlusPlus::from_entropy(),
            coster,
        ),
        solver,
    )
    .configure(|state| {
        state
            .param(solve_greedy(&tower, &words))
            .target_cost(coster.target)
    })
    .add_observer(
        observer::Observer,
        argmin::core::observers::ObserverMode::NewBest,
    )
    .run()?;
    println!();

    println!(
        "TOTAL SCORE: {}",
        score_solution(&tower, &res.state.best_param.unwrap())
    );

    Ok(())
}
