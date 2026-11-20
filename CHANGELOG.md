# Changelog

All notable changes to this project are grouped here.
Generated with git-cliff from Conventional Commits; `CHANGELOG.md` keeps the curated
Unreleased section between releases.


## [1.0.0] - 2026-11-20


### Added

- settings, logging and cli version

- **domain:** schema and half-open time windows

- **domain:** data contract validation engine

- **data:** bundle UCI bike sharing with manifest and verification

- **cli:** data and validate commands with DATA docs

- **domain:** shared feature pipeline and regression metrics

- **domain:** baseline, drift statistics and day-block bootstrap

- **domain:** model lifecycle state machine and domain events

- **domain:** promotion gates

- **domain:** retrain, promotion and probation policies

- **app:** ports, dto and pipeline with run ledger

- **app:** training and evaluation use cases

- **app:** promotion, rejection, rollback and bootstrap use cases

- **app:** scoring service and in-process gateway

- **storage:** sqlite migrations and repositories

- **models:** lightgbm trainer and replay clocks

- **mlflow:** tracker and alias-tag registry adapter

- **app:** monitoring snapshot service

- **drift:** evidently report adapter

- **app:** lifecycle tick use case with run ledger

- **bootstrap:** composition root

- **api:** request schemas and prometheus metrics

- **api:** model holder with atomic hot reload

- **api:** serving routes, request ids and app factory

- **serving:** http scoring gateway and http serving mode

- **cli:** serve and score commands

- **drill:** fault injection and the four scenario scripts

- **drill:** reproducible report model and performance summary

- **drill:** replay engine and golden expectations

- **cli:** bootstrap, train, registry, monitor and drill commands

- **scheduler:** replay scheduler with persisted cursor

- **cli:** environment doctor command

- **cli:** quickstart and replay cursor commands

- **cli:** scheduler start command

- **cli:** mlflow ui launcher

- **dashboard:** theme tokens and contrast checks

- **dashboard:** lifecycle view models

- **dashboard:** overview and monitoring views

- **dashboard:** registry, lifecycle and drill views

- **cli:** dashboard launcher and apptest smoke

- **dashboard:** loading spinner for page data

- **cli:** bootstrap at a simulated instant for replay workspaces

- **docker:** multi-stage image with uv and a non-root runtime

- **compose:** stack with healthchecks, hardening and a shared volume

- **install:** bash installer with prerequisites and uninstall

- **install:** PowerShell installer with prerequisites and uninstall

- **install:** reproducible compose bundle builder

- **docs:** generate results tables and curated drift reports

- **docs:** enforce links, ADRs, reports and results tables

- **docs:** capture real dashboard screenshots

- **docs:** assemble the hero gif from eight frames



### CI

- add lint and test skeleton workflow

- pin setup-uv to the published release tag

- full matrix, audit, secrets and compose smoke

- release workflow and compose bundle



### Changed

- **bootstrap:** share one replay source across the container



### Documentation

- **changelog:** record lifecycle, gates and policy work

- **changelog:** record application layer work

- **changelog:** record adapter layer work

- **changelog:** record serving API work

- **changelog:** record the drill engine and scenarios

- **changelog:** record scheduler, doctor and quickstart work

- **changelog:** record the control room dashboard

- document installers, CI and the security posture

- commit drill results and curated drift reports

- document the architecture and model card

- record registry, triggers, drift, scheduler and storage decisions

- add the operations runbook

- document the documentation gates and update the changelog

- readme in english and spanish

- screenshots and hero gif captured from the app



### Fixed

- **domain:** anchor PSI quantile edges on observed values

- **storage:** share the sqlite connection with serving worker threads

- **config:** treat an empty admin api key as unset



### Tests

- cover settings, logging and cli version

- **domain:** schema and time window contracts

- **domain:** data validation rules and verdicts

- **data:** bundled provenance and checksum refusal

- **cli:** data and validate command contract

- **cli:** pin a wide colour-free terminal for assertions

- **domain:** feature pipeline and regression metric contracts

- **domain:** baseline, PSI drift and bootstrap contracts

- **domain:** align baseline and bootstrap assertions with their contracts

- **domain:** keep numeric properties within floating-point tolerance

- **domain:** lifecycle state machine and event contracts

- **domain:** promotion gate and policy contracts

- **app:** pipeline, run ledger and dto contracts

- **app:** training and evaluation use case contracts

- **app:** promotion, rollback and bootstrap contracts

- **app:** scoring service and shared registry/prediction-log contracts

- **storage:** sqlite store and prediction-log contracts

- **models:** lightgbm trainer and clock contracts

- **mlflow:** tracker, registry and shared alias contracts

- bootstrap champion quality gate through real adapters

- **api:** serving integration through the real stack

- **cli:** render help without colour for cross-platform assertions

- **api:** serve the real stack through uvicorn on a loopback port

- drill goldens, end-to-end scenarios and quality gates

- **ui:** playwright smoke for the control room

- **ui:** ignore transient browser resource errors

- **dashboard:** tidy apptest imports

- **compose:** static stack contract

- **docker:** compose smoke and automated promotion check

- **install:** installer exit-code and PowerShell contracts

- **docs:** add documentation consistency contract


