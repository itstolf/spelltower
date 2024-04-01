use num_traits::ToPrimitive as _;

struct LogKV<'a, I>(Option<&'a I>, &'a argmin::core::KV);

impl<'a, I> log::kv::Source for LogKV<'a, I>
where
    I: argmin::core::State,
{
    fn visit<'kvs>(
        &'kvs self,
        visitor: &mut dyn log::kv::VisitSource<'kvs>,
    ) -> Result<(), log::kv::Error> {
        if let Some(state) = self.0 {
            for (k, v) in state.get_func_counts().iter() {
                visitor.visit_pair(log::kv::Key::from_str(k), log::kv::Value::from_display(v))?;
            }
            visitor.visit_pair(
                log::kv::Key::from_str("best_cost"),
                state.get_best_cost().to_f64().unwrap().into(),
            )?;
            visitor.visit_pair(
                log::kv::Key::from_str("cost"),
                state.get_cost().to_f64().unwrap().into(),
            )?;
            visitor.visit_pair(log::kv::Key::from_str("iter"), state.get_iter().into())?;
        }

        let mut kvs = self.1.kv.iter().collect::<Vec<_>>();
        kvs.sort_unstable_by_key(|(k, _)| &**k);

        for (k, v) in kvs {
            visitor.visit_pair(log::kv::Key::from_str(k), log::kv::Value::from_display(v))?;
        }
        Ok(())
    }
}

pub struct Observer;

impl<I> argmin::core::observers::Observe<I> for Observer
where
    I: argmin::core::State,
{
    fn observe_init(
        &mut self,
        name: &str,
        _state: &I,
        kv: &argmin::core::KV,
    ) -> Result<(), anyhow::Error> {
        log::logger().log(
            &log::RecordBuilder::new()
                .level(log::Level::Info)
                .target("argmin")
                .key_values(&LogKV::<I>(None, kv))
                .args(format_args!("{}", name))
                .build(),
        );
        Ok(())
    }

    fn observe_iter(&mut self, state: &I, kv: &argmin::core::KV) -> Result<(), anyhow::Error> {
        log::logger().log(
            &log::RecordBuilder::new()
                .level(log::Level::Info)
                .target("argmin")
                .key_values(&LogKV(Some(state), kv))
                .build(),
        );
        Ok(())
    }
}
