pub struct Coster {
    pub cost: for<'a> fn(tower: &'a ndarray::Array2<char>, param: &[Vec<(usize, usize)>]) -> f64,
    pub target: f64,
}

pub struct Annealer<'a> {
    tower: &'a ndarray::Array2<char>,
    root: &'a crate::words::Node,
    allow_leftovers: bool,
    rng: std::cell::RefCell<rand_xoshiro::Xoshiro256PlusPlus>,
    coster: &'static Coster,
}

impl<'a> Annealer<'a> {
    pub fn new(
        tower: &'a ndarray::Array2<char>,
        root: &'a crate::words::Node,
        allow_leftovers: bool,
        rng: rand_xoshiro::Xoshiro256PlusPlus,
        coster: &'static Coster,
    ) -> Self {
        Self {
            tower,
            root,
            allow_leftovers,
            rng: std::cell::RefCell::new(rng),
            coster,
        }
    }
}

impl<'a> argmin::core::CostFunction for Annealer<'a> {
    type Param = Vec<Vec<(usize, usize)>>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        if !self.allow_leftovers {
            let mut tower = self.tower.clone();
            for path in param {
                super::delete_path(&mut tower, path);
            }
            if tower.len() - tower.iter().filter(|&&x| x == '\0').count() != 0 {
                return Ok(std::f64::MAX);
            }
        }

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
        let mut rng = self.rng.borrow_mut();
        let mut solution = param.to_vec();
        for _ in 0..(temp.floor() as u64 + 1) {
            super::nudge_solution(&self.tower, &self.root, &mut solution, &mut *rng);
        }
        Ok(solution)
    }
}

fn longest_word(solution: &[Vec<(usize, usize)>]) -> usize {
    solution.iter().map(|v| v.len()).max().unwrap_or(0)
}

fn total_score(tower: &ndarray::Array2<char>, soultion: &[Vec<(usize, usize)>]) -> usize {
    let mut tower = tower.clone();
    let mut total_score = 0;
    for path in soultion {
        total_score += super::score_path(&tower, &path);
        super::delete_path(&mut tower, path);
    }
    if tower.iter().filter(|&&x| x == '\0').count() == tower.len() {
        total_score += 1000;
    }
    if super::is_almost_there(&tower) {
        total_score += 1000;
    }
    total_score
}

fn best_word_score(tower: &ndarray::Array2<char>, soultion: &[Vec<(usize, usize)>]) -> usize {
    let mut tower = tower.clone();
    let mut best_score = 0;
    for path in soultion {
        best_score = best_score.max(super::score_path(&tower, &path));
        super::delete_path(&mut tower, path);
    }
    best_score
}

pub const LONGEST_WORD: Coster = Coster {
    target: std::f64::NEG_INFINITY,
    cost: |_tower, solution| -(longest_word(solution) as f64),
};

pub const TOTAL_SCORE: Coster = Coster {
    target: std::f64::NEG_INFINITY,
    cost: |tower, solution| -(total_score(tower, solution) as f64),
};

pub const BEST_WORD: Coster = Coster {
    target: std::f64::NEG_INFINITY,
    cost: |tower, solution| -(best_word_score(tower, solution) as f64),
};

pub const FEWEST_WORDS: Coster = Coster {
    target: 1.0,
    cost: |_tower, solution| solution.len() as f64,
};
