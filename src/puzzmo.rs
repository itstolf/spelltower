const ENDPOINT: &str = "https://www.puzzmo.com/_api/prod/graphql";

const GQL_QUERY: &str = r#"query TodayScreenQuery($day: String) { todayPage(day: $day) { daily { isToday day puzzles { puzzle { puzzle game { slug } } } } } }"#;

#[derive(serde::Deserialize)]
#[serde(untagged, rename_all = "camelCase")]
enum Response {
    Data { data: response::Data },
    Errors { errors: Vec<response::Error> },
}

mod response {
    #[derive(serde::Deserialize, Debug)]
    #[serde(rename_all = "camelCase")]
    pub struct Error {
        pub message: String,
    }

    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Data {
        pub today_page: TodayPage,
    }

    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct TodayPage {
        pub daily: Daily,
    }

    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Daily {
        pub day: String,
        pub is_today: bool,
        pub puzzles: Vec<PuzzleWrapper>,
    }

    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct PuzzleWrapper {
        pub puzzle: Puzzle,
    }

    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Puzzle {
        pub puzzle: String,
        pub game: Game,
    }

    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Game {
        pub slug: String,
    }
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct Request {
    query: String,
    variables: std::collections::HashMap<String, serde_json::Value>,
}

pub struct Puzzle {
    pub day: String,
    pub is_today: bool,
    pub tower: crate::Tower,
}

#[derive(thiserror::Error, Debug)]
#[error("puzzmo: {errors:?}")]
pub struct Error {
    pub errors: Vec<response::Error>,
}

pub fn get_puzzle(day: Option<String>) -> anyhow::Result<Puzzle> {
    let client = reqwest::blocking::Client::new();

    let data = match client
        .post(ENDPOINT)
        .header("puzzmo-gameplay-id", "!")
        .json(&Request {
            query: GQL_QUERY.to_string(),
            variables: [("day".to_string(), day.into())].into(),
        })
        .send()?
        .error_for_status()?
        .json()?
    {
        Response::Data { data } => data,
        Response::Errors { errors } => {
            return Err(Error { errors }.into());
        }
    };

    let puzzle = data
        .today_page
        .daily
        .puzzles
        .into_iter()
        .find(|puzzle| puzzle.puzzle.game.slug == "spelltower")
        .map(|puzzle| puzzle.puzzle.puzzle)
        .ok_or_else(|| anyhow::anyhow!("could not find puzzle"))?;

    let mut puzzle_iter = puzzle.split("\n");
    let _ = puzzle_iter.next();

    let (w, h) = puzzle_iter.next().unwrap().split_once("x").unwrap();
    let w: usize = w.parse()?;
    let h: usize = h.parse()?;

    Ok(Puzzle {
        day: data.today_page.daily.day,
        is_today: data.today_page.daily.is_today,
        tower: ndarray::Array2::from_shape_vec(
            (h, w),
            puzzle_iter
                .flat_map(|row| row.chars())
                .take(w * h)
                .collect(),
        )?,
    })
}
