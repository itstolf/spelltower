pub struct Coster {
    pub cost: for<'a> fn(tower: &'a ndarray::Array2<char>, param: &[Vec<(usize, usize)>]) -> f64,
    pub target: f64,
}

pub struct Annealer<'a> {
    tower: &'a ndarray::Array2<char>,
    root: &'a crate::words::Node,
    rng: std::sync::Arc<std::sync::Mutex<rand_xoshiro::Xoshiro256PlusPlus>>,
    coster: &'static Coster,
}

impl<'a> Annealer<'a> {
    pub fn new(
        tower: &'a ndarray::Array2<char>,
        root: &'a crate::words::Node,
        rng: rand_xoshiro::Xoshiro256PlusPlus,
        coster: &'static Coster,
    ) -> Self {
        Self {
            tower,
            root,
            rng: std::sync::Arc::new(std::sync::Mutex::new(rng)),
            coster,
        }
    }
}

impl<'a> argmin::core::CostFunction for Annealer<'a> {
    type Param = Vec<Vec<(usize, usize)>>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok((self.coster.cost)(&self.tower, param))
    }
}

impl<'a> argmin::solver::simulatedannealing::Anneal for Annealer<'a> {
    type Param = Vec<Vec<(usize, usize)>>;
    type Output = Vec<Vec<(usize, usize)>>;
    type Float = f64;

    fn anneal(
        &self,
        param: &Self::Param,
        temp: Self::Float,
    ) -> Result<Self::Output, anyhow::Error> {
        let mut rng = self.rng.lock().unwrap();
        let mut solution = param.to_vec();
        for _ in 0..(temp.floor() as u64 + 1) {
            super::nudge_solution(&self.tower, &self.root, &mut solution, &mut *rng);
        }
        Ok(solution)
    }
}

fn longest_word<'a>(_tower: &'a ndarray::Array2<char>, solution: &[Vec<(usize, usize)>]) -> usize {
    solution.iter().map(|v| v.len()).max().unwrap_or(0)
}

fn total_score<'a>(tower: &'a ndarray::Array2<char>, soultion: &[Vec<(usize, usize)>]) -> usize {
    let mut tower = tower.clone();
    let mut total_score = 0;
    for path in soultion {
        total_score += super::score_path(&tower, &path);
        super::delete_path(&mut tower, path);
    }
    if tower.iter().filter(|&&x| x == '\0').count() == tower.len() {
        total_score += 1000;
    }
    total_score
}

fn best_word_score<'a>(
    tower: &'a ndarray::Array2<char>,
    soultion: &[Vec<(usize, usize)>],
) -> usize {
    let mut tower = tower.clone();
    let mut best_score = 0;
    for path in soultion {
        best_score = best_score.max(super::score_path(&tower, &path));
        super::delete_path(&mut tower, path);
    }
    best_score
}

fn letters_remaining<'a>(
    tower: &'a ndarray::Array2<char>,
    solution: &[Vec<(usize, usize)>],
) -> usize {
    let mut tower = tower.clone();
    for path in solution {
        super::delete_path(&mut tower, path);
    }
    tower.len() - tower.iter().filter(|&&x| x == '\0').count()
}

pub const LONGEST_WORD: Coster = Coster {
    target: std::f64::NEG_INFINITY,
    cost: |tower, solution| -(longest_word(tower, solution) as f64),
};

pub const TOTAL_SCORE: Coster = Coster {
    target: std::f64::NEG_INFINITY,
    cost: |tower, solution| -(total_score(tower, solution) as f64),
};

pub const BEST_WORD: Coster = Coster {
    target: std::f64::NEG_INFINITY,
    cost: |tower, solution| -(best_word_score(tower, solution) as f64),
};

pub const LETTERS_REMAINING: Coster = Coster {
    target: 0.0,
    cost: |tower, solution| letters_remaining(tower, solution) as f64,
};
