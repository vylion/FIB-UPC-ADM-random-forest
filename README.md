# About

My first Random Forest algorithm implementation for a classification problem, as part of a university delivery.

## Database

I use a star database as input for the classification. From each star I keep about 5 characteristics (magnitude, distance, luminosity, color, temperature), a label (the spectral class, which is what the classification that the random forest has to work out) and a display name (either a proper/popular name, an abbreviated name, or the ID in the database if the first two are missing).

The used database is the [HYG Data (version 3)](http://www.astronexus.com/hyg).

Since star data isn't precise and some stars could belong to more than one class, these are considered to belong to all of the possible ones, and when predicting the class, if a prediction of any of the possible classes for a star will be considered a success.

## Decision Tree

It's more or less agnostic to the fact that the entries to classify are stars. It receives a list of entries (a `dataset`) and a list of `fields`. Each `entry` must be an object with an `entry.label`, with the class value that the decission tree must figure out; and an `entry.data`, which is a list of values. The length of `entry.data` is expected to match the length of `fields`, and each value of `fields` must be the name for the value in the same position at `entry.data`.

The training set is a subset of the whole database, and the resulting `tree` is tested against the remaining entries.

## Random Forest

The random forest is built by bootstrapping the original training set, and then creating a tree for each bootstrapped instance of the `dataset`.