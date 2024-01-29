//! Sparse Graphical Model (SGM) implementation
//!
//! This module contains the implementation of Sparse Graphical Model (SGM) as
//! described in the paper "Sparse Graphical Memory for Robust Planning" by
//! Laskin et al. (2020).
//!
//! The SGM is implemented as a component that can be used in any off-policy
//! algorithm, by simply providing a method on top of a [`ReplayBuffer`].
//!
//! The SGM is used in the [`crate::agents::DDPG_SGM`] algorithm.
use {
    crate::{
        envs::TensorConvertible,
        components::ReplayBuffer,
    },
    anyhow::Result,
    ordered_float::OrderedFloat,
    petgraph::{
        dot::Dot,
        stable_graph::{
            NodeIndex,
            StableGraph,
        },
        Undirected,
    },
    serde::Serialize,
    std::{
        collections::HashMap,
        fmt::{
            Debug,
            Display,
        },
        hash::Hash,
    },
};


#[derive(Clone, Copy, Serialize)]
pub enum DistanceMode {
    True,
    Estimated,
}

impl Display for DistanceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceMode::True => write!(f, "True"),
            DistanceMode::Estimated => write!(f, "Estimated"),
        }
    }
}

/// Return a dotviz representation of the given graph.
pub fn dot<S: Debug>(graph: &StableGraph<S, OrderedFloat<f64>, Undirected>) -> String {
    format!("{:?}", Dot::new(graph)).to_string()
}


/// Adds a single node to the graph if it is TWC-consistent.
pub fn try_adding_node<S, D>(
    graph: &mut StableGraph<S, OrderedFloat<f64>, Undirected>,
    indices: &mut HashMap<S, NodeIndex>,
    s1: &S,
    d: D,
    maxdist: f64,
    tau: f64,
) -> Result<()>
where
    S: Clone + Eq + Hash + TensorConvertible,
    D: Fn(&S, &S) -> f64,
{
    // check if new node is TWC consistent
    let is_twc_consistent = graph
        .node_weights()
        .all(|s2| {
            let c_out = *graph.node_weights().map(|w|
                OrderedFloat((d(s1, w) - d(s2, w)).abs())
            ).max().unwrap();
            let c_in = *graph.node_weights().map(|w|
                OrderedFloat((d(w, s1) - d(w, s2)).abs())
            ).max().unwrap();

            c_out >= tau && c_in >= tau
        });

    if is_twc_consistent {
        // add node
        let i1 = graph.add_node(s1.clone());
        indices.insert(s1.clone(), i1);

        // add edges
        let mut edges_to_add = Vec::new();
        for s2 in graph.node_weights() {
            // no self edges
            if s1 == s2 {
                continue;
            }

            let d_out = d(s1, s2);
            let d_in = d(s2, s1);

            if d_out < maxdist && d_in < maxdist {
                edges_to_add.push((i1, indices[s2], d_out));
                edges_to_add.push((indices[s2], i1, d_in));
            }
        }
        for (a, b, weight) in edges_to_add {
            graph.add_edge(a, b, OrderedFloat(weight));
        }
    }

    is_twc_consistent.then_some(()).ok_or_else(|| anyhow::anyhow!("Not TWC consistent"))
}

impl ReplayBuffer {
    /// Construct a sparse graph from the replay buffer.
    ///
    /// The resulting graph can be seen as a tau-approximate, Q-irrelevant
    /// abstraction. This means that as we vary tau, we vary the sparsity of
    /// the graph in a way that is irrelevant for the Q-function, even in the
    /// goal-conditioned setting.
    ///
    /// # Arguments
    ///
    /// * `d` - The distance function.
    /// * `maxdist` - The maximum distance between two nodes in the graph.
    /// * `tau` - The tau parameter to vary the graph sparsity.
    pub fn construct_sgm<S, D>(
        &self,
        d: D,
        maxdist: f64,
        tau: f64,
    ) -> (
        StableGraph<S, OrderedFloat<f64>, Undirected>,
        HashMap<S, NodeIndex>,
    )
    where
        S: Clone + Eq + Hash + TensorConvertible,
        D: Fn(&S, &S) -> f64,
    {
        // initialize the SGM data structures
        let mut graph: StableGraph<S, OrderedFloat<f64>, Undirected> = StableGraph::default();
        let mut indices: HashMap<S, NodeIndex> = HashMap::new();

        // iterate over the set of nodes in the buffer
        for s1 in self.all_states::<S>().iter() {

            // always accept the first node
            if graph.node_count() == 0 {
                let i1 = graph.add_node(s1.clone());
                indices.insert(s1.clone(), i1);
            } else {
                let _ = try_adding_node(&mut graph, &mut indices, s1, &d, maxdist, tau);
            }
        }

        (graph, indices)
    }
}
